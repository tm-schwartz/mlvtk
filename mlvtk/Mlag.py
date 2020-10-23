# cspell: disable
import copy
import itertools
import os
import pathlib
import shutil
import sys
import logging

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow.python as tfp
from plotly.subplots import make_subplots
from sklearn.decomposition import FastICA, PCA

from .CheckpointCallback import CheckpointCallback

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
    Base class containing custom code for loss surface visualization using filter
    normalization. This currently supports functional or sequential models, not
    custom models.
"""


class Mlag:
    def __init__(self, model, msaver_path, verbose):
        self.alphas = None
        self.betas = None
        self.testdat = None  # test data set used to calculate loss vals
        self.loss_df = None  # pandas data frame containing loss vals
        self.i_data = None  # interpolated loss vals
        self.xdir = None  # x values of optimizer path
        self.ydir = None  # y values of optimizer path
        self.evr = None  # explained variance ratio
        self.loss = None  # loss function
        self.overwrite = None
        self.opt = None
        self.msave_path = pathlib.Path(msaver_path)  # path to save directory
        self._compatible = False  # are model checkpoints tf compatible
        self._fit = model.fit
        self._compile = model.compile
        self._type = model.__class__
        self._logger = logging.getLogger("MLAG")
        self.verbose = verbose
        self.pca_dirs = None
        physical_devices = tf.config.list_physical_devices("GPU")

        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except BaseException:
                # Invalid device or cannot modify virtual devices once
                # initialized.
                pass

    def _tf_compatible(self):
        """
        Check if model checkpoints are compatible with tensorflow.keras
        loading. If not, then change `class_name` to be compatible.
        """

        self.msave_path = pathlib.Path(self.msave_path)

        if self._type == tfp.keras.engine.functional.Functional:
            replacement = (
                b'"class_name": "ModelVFunctional"',
                b'"class_name": "Functional"',
            )
        elif self._type == tf.keras.Sequential:
            replacement = (
                b'"class_name": "ModelVSequential"',
                b'"class_name": "sequential"',
            )
        for f in self.msave_path.glob(r"model_[0-9]*"):
            hf = h5py.File(f, "r+")
            hf.attrs.modify(
                "model_config",
                hf.attrs.get("model_config").replace(*replacement),
            )
            hf.close()
        self._compatible = True

    def compile(self, *args, **kwargs):
        """
        Compile model. Need to know what optimizer and loss function are in
        order to evaluate model later.

        Args:
            *args, **kwargs: positional and keyword arguments passed to model.compile

        Returns:
            model.compile(*args, **kwargs)
        """

        if kwargs.get("optimizer") is not None:
            self.opt = kwargs.get("optimizer").__module__.split(".")[-1]

        elif len(args):
            self.opt = args[0]

        else:
            self.opt = "rmsprop"

        # compatibility with tf.keras.optimizer.get(...)
        if self.opt == "gradient_descent":
            self.opt = "SGD"
        self.loss = kwargs.get("loss") if kwargs.get("loss") is not None else args[1]

        self._compile(*args, **kwargs)
        self._is_compiled = True

    def fit(self, *args, **kwargs):
        """
        Call model.fit. Create and possibly remove old checkpoint
        save directory. If old one is reused, can cause errors when
        calculating path of optimizer if early stopping or a different
        number of epochs are used.

        Also parse `callbacks` if present, and append
        `CheckpointCallback` to list of callbacks.
        If not present, then create list `callbacks` containing
        `CheckpointCallback`.

        Finally, verify that `validation_data` has been passed. Currently
        need seperate val data for calculating loss values.

        Args:
            *args, **kwargs: positional and keyword arguments to be passed
            to model.fit.
                - `validation_data` must be passed as either positional or
                    keyword.
        Returns:
            model.fit(*args, **kwargs)

        Raises:
            NotImplementedError: if `validation_data` is not passed.
        """

        self.msave_path = pathlib.Path(self.msave_path)

        if (
            self.msave_path.exists()
            and self.msave_path.is_dir()
            and self.msave_path.lstat().st_size
        ):

            if self.overwrite is not None:
                if self.overwrite:
                    shutil.rmtree(self.msave_path)
                elif self.overwrite == False:
                    self.msave_path = input("Please enter new save folder path ")
                    self.msave_path = pathlib.Path(self.msave_path)

            else:
                overwrite = input(
                    f"{self.msave_path} is not empty, overwrite? (this message can be silenced by setting `overwrite=True/False`)"
                )
                while True:
                    if overwrite in ["no", "n"]:
                        self.msave_path = input("Please enter new save folder path ")
                        self.msave_path = pathlib.Path(self.msave_path)
                        break
                    elif overwrite in ["yes", "y"]:
                        shutil.rmtree(self.msave_path)
                        break
                    overwrite = input("please enter: yes/no/y/n")

        if not self.msave_path.is_dir():
            self.msave_path.mkdir(parents=True)

        if kwargs.get("callbacks"):
            # TODO: add overwrite option
            kwargs["callbacks"].append(CheckpointCallback(path=self.msave_path))
        else:
            kwargs["callbacks"] = [CheckpointCallback(path=self.msave_path)]
        if kwargs.get("validation_data") is None and len(args) < 8:
            raise NotImplementedError("Need precomputed `validation_data`")
        self.testdat = kwargs.get("validation_data") if len(args) < 8 else args[7]
        return self._fit(*args, **kwargs)

    def _new_model(self):
        """ create a new model for evaluation """

        config = self.get_config()
        if self._type == tfp.keras.engine.functional.Functional:
            config["class_name"] = "Functional"
            new_mod = tf.keras.Model.from_config(config)
        elif self._type == tf.keras.Sequential:
            config["class_name"] = "sequential"
            new_mod = tf.keras.Sequential.from_config(config)
            new_mod.build(input_shape=self.layers[0].input_shape)

        new_mod.compile(optimizer=self.opt, loss=self.loss, run_eagerly=False)

        return new_mod

    def _evaluate_batch_size(self):
        if not isinstance(self.testdat, tf.data.Dataset):
            batch_size = self.testdat[0].shape[0]
        else:
            batch_size = None
        return batch_size

    def _calculate_loss(self, alpha_size, beta_size, ext, random_dirs):
        """
          Create pandas dataframe containing loss values found on loss surface
          of model. If `self.alphas` and `self.betas` are centered at 0, then
          (0,0) represents final loss of model on `validation_data`.

          -5   -4   -3    -2    -1    0    1    2    3    4    5   <- betas
          -------------------------------------------------------   _ alphas
        -5|                                                     |  v
          |                  _________                          |
          |          _______/   ....  \  <-- loss surface       |
        -4|       __|..................\                        |
          |      /....., , .............---__                   |
          |     /......., ,..................\                  |
        -3|    |........., ,..................\                 |
          |    |.........., ,..................-_               |
          |    |..........,,,....................\              |
        -2|    \..........,,,.....................\             |
          |     \..........,,.....................|             |
          |      \.........,,,....................|             |
        -1|       \..........,,....,,,,,,, ,......|             |
          |        \.........., ,,,,******, ,......--_          |
          |         \...........,,  **  ****,.........|         |
         0|      ____\............,*  +  *** ,........|         |  + -> final
          |     /..................,*    ***,.........|         |       loss
          |    /....................,******,..........|         |       of
         1|   /......................,,,,,,.........../         |       model
          |  /......................................./          |
          | /......................................./           |
         2| |....................................../            |
          | |...................................../             |
          | |.................................._--              |
         3| \................................./                 |
          |  \_----____.......... ___________/                  |
          |            \_________/                              |
         4|                                                     |
          |                                                     |
          |                                                     |
         5|                                                     |
          -------------------------------------------------------
        """

        self.msave_path = pathlib.Path(self.msave_path)

        if self.xdir is None or self.ydir is None:
            self.gen_path()

        weights = np.asarray(self.get_weights())

        if self.xdir.size == 1:
            self.alphas = np.linspace(-5, 5, num=alpha_size, dtype=np.float32)

            self.betas = np.linspace(-5, 5, num=beta_size, dtype=np.float32)

        elif self.xdir.size > 1:
            if ext == "std":
                diff_alphas = (self.xdir.mean() + self.xdir.std()) - (
                    self.xdir.mean() - self.xdir.std()
                ) * 0.5
                diff_betas = (self.ydir.mean() + self.ydir.std()) - (
                    self.ydir.mean() - self.ydir.std()
                ) * 0.5

            elif ext == 1:
                diff_alphas = diff_betas = 1

            else:
                diff_alphas, diff_betas = ext

            self.alphas = np.linspace(
                self.xdir.min() - diff_alphas,
                self.xdir.max() + diff_alphas,
                num=alpha_size,
                dtype=np.float32,
            )

            self.betas = np.linspace(
                self.ydir.min() - diff_betas,
                self.ydir.max() + diff_betas,
                num=beta_size,
                dtype=np.float32,
            )

        # For readability, make local vars
        alph = self.alphas[None, :, None]  # [[[`val`], [`val`],...]]
        bet = self.betas[:, None, None]  # [[[`val`]], [[`val`]]...]
        bet_zeros = np.zeros_like(bet)
        alph_zeros = np.zeros_like(alph)

        # in order to construct cartesian product:
        # take advantage of broadcasting with tf.add.
        # example:::
        # alphas.shape -> (50,)
        # betas.shape -> (50,)
        # alph.shape/alph_zeros.shape -> (1, 50, 1)
        # bet.shape/bet_zeros.shape -> (50, 1, 1)
        #
        # By doing sum_alph = tf.add(alph, bet_zeros) and sum_bet = tf.add(bet, alph_zeros)
        # we get two tensors of shape (50, 50, 1).
        # Then we tf.concat([sum_alph, sum_bet], axis=2).
        # By doing it along axis=2, we get pairs of values
        # in shape (50, 50, 2). Finally, we use tf.reshape
        # to get final cartesian product in result.shape = (2500, 2),
        # so then result[i] == [`val_i_0`, `val_i_1`]
        # adapted from https://stackoverflow.com/a/50195230
        paramlist = np.reshape(
            np.concatenate([alph + bet_zeros, alph_zeros + bet], axis=2),
            (self.alphas.shape[0] * self.betas.shape[0], 2),
        )

        def reshape(values):
            s = 0
            for w in weights:
                yield np.reshape(values[s : s + w.size], w.shape)
                s += w.size

        def filter_normalize():
            """
            if random_dirs:
                delta = np.array(
                    [np.reshape(np.random.randn(ww.size), ww.shape) for ww in weights],
                    dtype="object",
                )
                etta = np.array(
                    [np.reshape(np.random.randn(ww.size), ww.shape) for ww in weights],
                    dtype="object",
                )

            else:
            """
            pd, pe = self.pca_dirs
            delta = np.array([pca_delta for pca_delta in reshape(pd)])
            etta = np.array([pca_etta for pca_etta in reshape(pe)])
            bn = filter(
                lambda layer: "batch_normalization" in layer.name,
                self.layers,
            )
            if bn:
                for layer in bn:
                    i = self.layers.index(layer)
                    delta[i] = np.zeros((delta[i].shape))
                    etta[i] = np.zeros((etta[i].shape))

            def normalize_filter(fw):
                f, w = fw
                return f * np.array([(np.linalg.norm(w) / (np.linalg.norm(f) + 1e-10))])

            normalized_filter_array = np.array(
                [
                    np.array(
                        [
                            d0 * a + d1 * b
                            for d0, d1 in zip(
                                map(normalize_filter, zip(delta, weights)),
                                map(normalize_filter, zip(etta, weights)),
                            )
                        ],
                        dtype="object",
                    )
                    for a, b in paramlist
                ],
                dtype="object",
            )

            optimizer_filter_array = np.array(
                [
                    np.array(
                        [
                            d0 * x + d1 * y
                            for d0, d1 in zip(
                                map(normalize_filter, zip(delta, weights)),
                                map(normalize_filter, zip(etta, weights)),
                            )
                        ],
                        dtype="object",
                    )
                    for x, y in zip(self.xdir, self.ydir)
                ],
                dtype="object",
            )

            return (
                normalized_filter_array.reshape(
                    self.alphas.shape[0], self.betas.shape[0], len(weights)
                ),
                optimizer_filter_array,
            )

        def filter_generator():
            for ix, iy in itertools.product(
                [*range(self.alphas.size)], [*range(self.betas.size)]
            ):
                yield normalized_filter_array[ix, iy]

        def zdir_filter_generator():
            for i in range(self.xdir.size):
                yield optimizer_filter_array[i]

        def _calc_weights(data):
            return np.array(
                [mw + data[k] for k, mw in enumerate(weights)],
                dtype="object",
            )

        normalized_filter_array, optimizer_filter_array = filter_normalize()

        def get_losses():

            new_mod = self._new_model()
            batch_size = self._evaluate_batch_size()

            direction_losses = np.zeros((self.xdir.size))

            if self.verbose:
                self._logger.setLevel(logging.INFO)
            self._logger.info("Calculating z-values")
            prog_bar = tf.keras.utils.Progbar(
                self.xdir.size,
                width=30,
                verbose=self.verbose,
                interval=0.05,
                stateful_metrics=None,
                unit_name="step",
            )
            #            with tf.device("/:GPU:0"):
            for step, new_weights in enumerate(
                map(
                    _calc_weights,
                    zdir_filter_generator(),
                )
            ):

                new_mod.set_weights(new_weights)

                l = new_mod.evaluate(
                    self.testdat,
                    use_multiprocessing=True,
                    batch_size=batch_size,
                    verbose=0,
                )

                direction_losses[step] = l

                prog_bar.update(step + 1)

            self.zdir = direction_losses

            losses = np.zeros((self.alphas.size * self.betas.size))

            new_mod = self._new_model()

            self._logger.info("Calculating surface values")
            prog_bar = tf.keras.utils.Progbar(
                self.alphas.size * self.betas.size,
                width=30,
                verbose=self.verbose,
                interval=0.05,
                stateful_metrics=None,
                unit_name="step",
            )
            # with tf.device("/:GPU:0"):
            for step, new_weights in enumerate(
                map(
                    _calc_weights,
                    filter_generator(),
                )
            ):

                new_mod.set_weights(new_weights)

                l = new_mod.evaluate(
                    self.testdat,
                    use_multiprocessing=True,
                    batch_size=batch_size,
                    verbose=0,
                )

                losses[step] = l
                prog_bar.update(step + 1)

            return losses

        raw_losses = get_losses()

        self.loss_df = pd.DataFrame(
            data=raw_losses.reshape(self.alphas.size, self.betas.size),
            index=self.alphas,
            columns=self.betas,
        )

    def gen_path(self):
        self.msave_path = pathlib.Path(self.msave_path)
        assert (
            self.msave_path.is_dir()
        ), 'Could not find model save path. Check "msave_path" is set correctly'

        if not self._compatible:
            self._tf_compatible()
        files = [
            file_path
            for file_path in sorted(
                self.msave_path.glob(r"model_[0-9]*"),
                key=lambda x: int(x.parts[-1].split("_")[-1][:-3]),
            )
        ]
        final_model = tf.keras.models.load_model(files[-1])
        theta_n = final_model.get_weights()
        raw_weights = [
            tf.keras.models.load_model(model_file).get_weights()
            for model_file in files[:-1]
        ]
        weight_differences_btw_epochs = [
            [theta_i - theta_n_i for theta_i, theta_n_i in zip(theta, theta_n)]
            for theta in raw_weights
        ]

        # TODO: rename
        def tensorlist_to_tensor_tf(weights):
            return np.concatenate([w.flatten() if w.ndim > 1 else w for w in weights])

        def npvec_to_tensorlist_tf(direction, params):
            w2 = copy.deepcopy(params)
            idx = 0
            for i, w in enumerate(w2):

                w2[i] = direction[idx : idx + w.size]
                w2[i] = w2[i].flatten()
                idx += w.size
            return np.concatenate(w2)

        def project_1d_tf(w, d):

            assert len(w) == len(d), "dimension does not match for w and d"
            return np.dot(np.array(w), d) / np.linalg.norm(d)

        def project_2d_tf(d, dx, dy):

            x = project_1d_tf(d, dx)
            y = project_1d_tf(d, dy)
            return x, y

        # ica = FastICA(n_components=2, fun="logcosh", max_iter=800)
        pca = PCA(n_components=2)
        T0 = np.array(
            [tensorlist_to_tensor_tf(i) for i in weight_differences_btw_epochs]
        )
        xdir, ydir = [], []

        if not np.any(T0):
            self.xdir = np.array([0])
            self.ydir = np.array([0])
            self._logger.setLevel(logging.WARNING)
            self._logger.warning("No weight change between epochs")
            self.evr = [pd.NA, pd.NA]
            return

        pca.fit(T0)
        pca_1 = pca.components_[0]
        pca_2 = pca.components_[1]
        self.pca_dirs = [pca_1, pca_2]

        for ep in T0:
            xd, yd = project_2d_tf(
                ep,
                npvec_to_tensorlist_tf(pca_1, theta_n),
                npvec_to_tensorlist_tf(pca_2, theta_n),
            )
            xdir.append(xd)
            ydir.append(yd)

        self.xdir = np.array(xdir)
        self.ydir = np.array(ydir)
        self.evr = pca.explained_variance_ratio_

    def surface_plot(
        self,
        title_text=None,
        save_file=None,
        return_traces=False,
        alpha_size=50,
        beta_size=50,
        ext=1,
        show_arrow=False,
        recalc=False,
        random_dirs=False,
    ):

        self.msave_path = pathlib.Path(self.msave_path)

        if np.any(self.loss_df) is None or recalc:
            self._calculate_loss(alpha_size, beta_size, ext, random_dirs)

        surface_trace = go.Surface(
            x=self.loss_df.index,
            y=self.loss_df.columns,
            z=self.loss_df.values,
            opacity=0.9,
            coloraxis="coloraxis",
            lighting=dict(ambient=0.6, roughness=0.9, diffuse=0.5, fresnel=2),
            name=f"loss surface",
        )

        scatter_trace = go.Scatter3d(
            x=self.xdir,
            y=self.ydir,
            z=self.zdir,
            marker=dict(symbol="circle", size=2, color="rgba(256, 0, 0, 80)"),
            line=dict(color="darkblue", width=2),
            showlegend=True,
            name=f"{self.opt} path",
        )

        title = (
            f"Component 1 EVR: {self.evr[0]:.4f}, Component 2 EVR: {self.evr[1]:.4f}"
        )

        # TODO: refactor title

        if title_text:
            if title_text[0] == "+":
                title = dict(
                    text=f"{title_text[1:]}, Component 1 EVR: {self.evr[0]:.4f}, Component 2 EVR: {self.evr[1]:.4f}",
                    x=0.5,
                )
            else:
                title = dict(text=title_text, x=0.7)
        else:
            title = dict(text=title, x=0.7)

        fig = go.Figure(data=[surface_trace, scatter_trace])
        fig.update_layout(
            title=title,
            autosize=False,
            width=1200,
            height=900,
            margin=dict(l=10),
            bargap=0.2,
            coloraxis=dict(
                colorscale="haline_r",
                colorbar=dict(title="Loss Surface Value", len=0.95),
            ),
        )
        if show_arrow:
            fig.update_layout(
                scene=dict(
                    annotations=[
                        dict(
                            showarrow=True,
                            x=self.xdir[0],
                            y=self.ydir[0],
                            z=self.zdir[0],
                            text="Start",
                        ),
                        dict(
                            showarrow=True,
                            x=self.xdir[-1],
                            y=self.ydir[-1],
                            z=self.zdir[-1],
                            text="End",
                        ),
                    ]
                )
            )

        if save_file is not None:
            fig.write_html(f"{save_file}.html")
        # fig.write_image(f"{save_file}.svg")

        if return_traces:
            return [surface_trace, scatter_trace]

        return fig

    def _interpolate(self):
        _test = np.zeros(self.alphas.size)
        f_list = sorted(
            self.msave_path.glob(r"model_[0-9]*"),
            key=lambda x: int(x.parts[-1].split("_")[-1][:-3]),
        )
        mod_0 = tf.keras.models.load_model(f_list[0])
        mod_1 = tf.keras.models.load_model(f_list[-1])
        w_0 = mod_0.get_weights()
        w_1 = mod_1.get_weights()
        del mod_0

        batch_size = self._evaluate_batch_size()

        pb = tf.keras.utils.Progbar(self.alphas.size, unit_name="alpha")

        for i, alpha in enumerate(self.alphas):
            theta_a = [
                theta_0 + theta_1
                for theta_0, theta_1 in zip(
                    [(1 - alpha) * w for w in w_0], [alpha * w for w in w_1]
                )
            ]  # read from msave_path
            mod_1.set_weights(theta_a)
            res = mod_1.evaluate(self.testdat, batch_size=batch_size, verbose=0)
            _test[i] = res[0]
            pb.update(i + 1)
        self.i_data = _test  # change returned val

    def interp_plot(self, *args, **kwargs):
        if np.any(self.i_data) is None:
            self._interpolate()
        fig = go.Figure(data=go.Scatter(x=self.alphas, y=self.i_data))
        fig.update_layout(*args, **kwargs)
        return fig
