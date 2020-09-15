# cspell: disable
import copy
import itertools
import os
import pathlib
import shutil

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from .CheckpointCallback import CheckpointCallback

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
    Base class containing custom code for loss surface visualization using filter
    normalization. This currently supports functional or sequential models, not
    custom models.
"""


class Mlag:
    def __init__(self, model, msaver_path):
        self.alphas = np.linspace(-3, 3, 35) 
        self.betas = np.linspace(-3, 3, 35)
        self.testdat = None  # test data set used to calculate loss vals
        self.loss_df = None  # pandas data frame containing loss vals
        self.i_data = None  # interpolated loss vals
        self.xdir = None  # x values of optimizer path
        self.ydir = None  # y values of optimizer path
        self.evr = None  # explained variance ratio
        self.loss = None  # loss function
        self.opt = None
        self.msave_path = pathlib.Path(msaver_path)  # path to save directory
        self._compatible = False  # are model checkpoints tf compatible
        self._fit = model.fit
        self._compile = model.compile
        self._type = model.__class__

        physical_devices = tf.config.list_physical_devices('GPU')

        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

    def _tf_compatible(self):

        """
            Check if model checkpoints are compatible with tensorflow.keras
            loading. If not, then change `class_name` to be compatible.
        """

        self.msave_path = pathlib.Path(self.msave_path)

        if self._type == tf.python.keras.engine.functional.Functional:
            replacement = (
                b'"class_name": "ModelVFunctional"',
                b'"class_name": "Functional"',
            )
        elif self._type == tf.keras.Sequential:
            replacement = (
                b'"class_name": "ModelVSequential"',
                b'"class_name": "sequential"',
            )
        for f in self.msave_path.glob(r'model_[0-9]*'):
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
        self.opt = (
            kwargs.get("optimizer") if kwargs.get("optimizer") != None else args[0]
        ).__module__.split(".")[-1]
        # compatibility with tf.keras.optimizer.get(...)
        if self.opt == 'gradient_descent':
            self.opt = 'SGD'
        self.loss = kwargs.get("loss") if kwargs.get("loss") != None else args[1]

        self._compile(*args, **kwargs)
        self._is_compiled = True

        #return self._compile(*args, **kwargs)

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

        if self.msave_path.exists() and self.msave_path.is_dir() and self.msave_path.lstat().st_size:

            overwrite = input(f"{self.msave_path} is not empty, overwrite?")
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
        if kwargs.get("validation_data") == None and len(args) < 8:
            raise NotImplementedError("Need precomputed `validation_data`")
        self.testdat = kwargs.get("validation_data") if len(args) < 8 else args[7]
        return self._fit(*args, **kwargs)

    def _calculate_loss(self):
        """
            Create pandas dataframe containing loss values found on loss surface
            of model. If `self.alphas` and `self.betas` are centered at 0, then
            (0,0) represents final loss of model on `validation_data`.

            -5   -4   -3    -2    -1    0    1    2    3    4    5   <- alphas
            -------------------------------------------------------   _ betas
          -5|                                                     |  v 
            |                                                     |
            |                                                     |
          -4|                                                     |
            |                                                     |
            |                                                     |
          -3|                                                     |
            |                                                     |
            |                                                     |
          -2|                                                     |
            |                                                     |
            |                                                     |
          -1|                                                     |
            |                                                     |
            |                                                     |
           0|                           +                         |  + -> final
            |                                                     |       loss
            |                                                     |       of
           1|                                                     |       model
            |                                                     |
            |                                                     |
           2|                                                     |
            |                                                     |
            |                                                     |
           3|                                                     |
            |                                                     |
            |                                                     |
           4|                                                     |
            |                                                     |
            |                                                     |
           5|                                                     |
            -------------------------------------------------------
        """
        
        self.msave_path = pathlib.Path(self.msave_path)

        if self.xdir == None or self.ydir == None:
            self.gen_path()

        weights = np.asarray(self.get_weights())

        def filter_normalize():
            delta = np.array(
                [np.reshape(np.random.randn(ww.size), ww.shape) for ww in weights],
                dtype="object",
            )
            etta = np.array(
                [np.reshape(np.random.randn(ww.size), ww.shape) for ww in weights],
                dtype="object",
            )

            bn = filter(
                    lambda layer: layer.name == "batch_normalization",
                    self.layers,
                )
            if bn:
                for layer in bn:
                    i = self.layers.index(layer)
                    delta[i] = np.zeros((delta[i].shape))
                    etta[i] = np.zeros((etta[i].shape))
            paramlist = itertools.product(self.alphas, self.betas)

            def normalize_filter(fw):
                f, w = fw
                return f * np.array([(np.linalg.norm(w) / (np.linalg.norm(f) + 1e-10))])

            result = np.array(
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

            return result.reshape(
                self.alphas.shape[0], self.betas.shape[0], len(weights)
            )

        def gen_filter_final():
            for ix, iy in itertools.product(
                [*range(self.alphas.shape[0])], [*range(self.betas.shape[0])]
            ):
                yield (ix, iy, filters[ix, iy])

        def _calc_weights(data):
            return (
                data[0],
                data[1],
                np.array(
                    [mw + data[2][k] for k, mw in enumerate(weights)],
                    dtype="object",
                ),
            )

        def _calc_loss(w):
            new_mod.set_weights(w)
            return new_mod.evaluate(
                    self.testdat, use_multiprocessing=True, verbose=0
                )

        config = self.get_config()

        if self._type == tf.python.keras.engine.functional.Functional:
            config["class_name"] = "Functional"
            new_mod = tf.keras.Model.from_config(config)
        elif self._type == tf.keras.Sequential:
            config["class_name"] = "sequential"
            new_mod = tf.keras.Sequential.from_config(config)
            new_mod.build(input_shape=self.layers[0].input_shape)

        new_mod.compile(optimizer=self.opt, loss=self.loss)

        filters = filter_normalize()

        df = pd.DataFrame(index=self.alphas, columns=self.betas)

        prog_bar = tf.keras.utils.Progbar(
            self.alphas.size * self.betas.size,
            width=30,
            verbose=1,
            interval=0.05,
            stateful_metrics=None,
            unit_name="step",
        )
        with tf.device('/:GPU:0'):        
            for step, new_weights in enumerate(map(_calc_weights, gen_filter_final())):
                df.iloc[new_weights[0], new_weights[1]] = _calc_loss(new_weights[2])
                prog_bar.update(step + 1)
        self.loss_df = df

    def gen_path(self, res=1):
        self.msave_path = pathlib.Path(self.msave_path)
        assert self.msave_path.is_dir(), 'Could not find model save path. Check "msave_path" is set correctly'


        if not self._compatible:
            self._tf_compatible()
        files = [
            file_path for file_path in sorted(
                self.msave_path.glob(r'model_[0-9]*'),
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

        pca = PCA(n_components=2)
        T0 = [tensorlist_to_tensor_tf(i) for i in weight_differences_btw_epochs]
        pca.fit(np.array(T0))
        pca_1 = pca.components_[0]
        pca_2 = pca.components_[1]

        xdir, ydir = [], []
        for ep in T0:
            xd, yd = project_2d_tf(
                ep,
                npvec_to_tensorlist_tf(pca_1, theta_n),
                npvec_to_tensorlist_tf(pca_2, theta_n),
            )
            xdir.append(xd)
            ydir.append(yd)

        self.xdir = xdir
        self.ydir = ydir
        self.evr = pca.explained_variance_ratio_

        # to implement
        #self.alphas = np.concatenate([self.alphas, self.xdir])
        #self.betas = np.concatenate([self.betas, self.ydir])
        #self.alphas.sort()
        #self.betas.sort()



    def surface_plot(self, title_text=None, save_file=None,
            approximate_model_path=True):

        self.msave_path = pathlib.Path(self.msave_path)

        if np.any(self.loss_df) == None:
            self._calculate_loss()

        if self.xdir == None or self.ydir == None:
            self.gen_path()

        # TODO: fix title of plot
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"is_3d": True}]],
            subplot_titles=[
                f"Component 1 EVR: {self.evr[0]:.4f}, Component 2 EVR: {self.evr[1]:.4f}"
            ],
        )

        if approximate_model_path:

          xs = [
              self.loss_df.index[i]
              if i < self.loss_df.index.shape[0]
              else self.loss_df.index[i - 1]
              for i in np.digitize(self.xdir, self.loss_df.index, right=True)
          ]
          ys = [
              self.loss_df.columns[i]
              if i < self.loss_df.columns.shape[0]
              else self.loss_df.columns[i - 1]
              for i in np.digitize(self.ydir, self.loss_df.columns, right=True)
          ]

          zs = [self.loss_df.loc[x, y] for x, y in zip(xs, ys)]

        else:
            raise NotImplementedError()
            # zs = [self.loss_df.loc[x, y] for x, y in zip(self.xdir, self.ydir)]

        fig.add_trace(
            go.Surface(
                x=self.loss_df.index,
                y=self.loss_df.columns,
                z=self.loss_df.values,
                opacity=0.9,
                showscale=True,
                lighting=dict(ambient=0.6, roughness=0.9, diffuse=0.5, fresnel=2),
                colorscale="haline_r",
                colorbar=dict(lenmode="pixels", len=400),
            ),
            row=1,
            col=1,
        )
        fig.add_scatter3d(
            x=self.xdir,
            y=self.ydir,
            z=zs,
            marker=dict(size=2, color="red"),
            line=dict(color="darkblue", width=2),
            showlegend=True,
            name="opt path",
            row=1,
            col=1,
        )

        if title_text:
            title = dict(text=title_text, x=0.7)
        else:
            title = None

        fig.update_layout(
            title=title,
            autosize=False,
            width=1200,
            height=900,
            margin=dict(l=10),
            bargap=0.2,
        )
        fig.show()
        if save_file is not None:
            fig.write_html(f"{save_file}.html")
    
    def _interpolate(self):
        _test = np.zeros(self.alphas.shape[0])
        f_list = sorted(
                self.msave_path.glob(r'model_[0-9]*'),
                key=lambda x: int(x.parts[-1].split("_")[-1][:-3]),
            )
        mod_0 = tf.keras.models.load_model(f_list[0])
        mod_1 = tf.keras.models.load_model(f_list[-1])
        w_0 = mod_0.get_weights()
        w_1 = mod_1.get_weights()
        del mod_0

        pb = tf.keras.utils.Progbar(self.alphas.shape[0], unit_name='alpha')
        for i, alpha in enumerate(self.alphas):
            theta_a = [theta_0 + theta_1 for theta_0, theta_1 in
                    zip([(1-alpha)* w for w in w_0], [alpha *
                        w for w in w_1])] # read from msave_path
            mod_1.set_weights(theta_a)
            # mod_1.compile(optimizer=self.optimizer, loss=self.loss,
            # metrics=['accuracy'])
            res = mod_1.evaluate(self.testdat, verbose=0)
            _test[i] = res[0]
            pb.update(i + 1)
        self.i_data = _test # change returned val
    
    def interp_plot(self, *args, **kwargs):
        if np.any(self.i_data) == None:
            self._interpolate() 
        fig = go.Figure(data=go.Scatter(x=self.alphas, y=self.i_data))
        fig.update_layout(
               *args, **kwargs)
        fig.show()
