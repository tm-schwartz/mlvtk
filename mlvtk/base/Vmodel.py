import pathlib
import shutil
import typing

from typing import Union, Optional

import tensorflow as tf
from tensorflow.python.keras.engine.data_adapter import train_validation_split
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.keras import Sequential, Model

from .callbacks.CheckpointCallback import CheckpointCallback
from .normalize.FilterNorm import normalizer
from .normalize.CalcTrajectory import CalcTrajectory
from .plot import plotter


class Vmodel:
    def __init__(
        self,
        model: Optional[Union[Functional, Sequential, Model]] = None,
        checkpoint_path: Union[pathlib.Path, str] = "vwd",
        verbose: int = 0,
        inputs=None,
        outputs=None,
    ):

        if model is None:
            if inputs is not None and outputs is not None:
                self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        elif isinstance(model, list):
                self.model = tf.keras.Sequential(model)
        else:
            self.model = model

        self.verbose = verbose
        self.overwrite = False
        self.checkpoint_path: typing.Union[pathlib.Path, str] = pathlib.Path(
            checkpoint_path
        )

    def _get_cpoint_path(self) -> pathlib.Path:
        return pathlib.Path(self.checkpoint_path)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

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

        return self.__getattr__("compile")(*args, **kwargs)

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
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
        need separate val data for calculating loss values.

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

        if (
            self._get_cpoint_path().exists()
            and self._get_cpoint_path().is_dir()
            and self._get_cpoint_path().lstat().st_size
        ):

            if self.overwrite is not None:
                if self.overwrite:
                    shutil.rmtree(self._get_cpoint_path())
                elif self.overwrite == False:
                    self.checkpoint_path = input("Please enter new save folder path ")
            else:
                overwrite = input(
                    f"{self._get_cpoint_path()} is not empty, overwrite? (this message can be silenced by setting `overwrite=True/False`)"
                )
                while True:
                    if overwrite in ["no", "n"]:
                        self.checkpoint_path = input(
                            "Please enter new save folder path "
                        )
                        break
                    elif overwrite in ["yes", "y"]:
                        shutil.rmtree(self._get_cpoint_path())
                        break
                    overwrite = input("please enter: yes/no/y/n")

        if not self._get_cpoint_path().is_dir():
            self._get_cpoint_path().mkdir(parents=True)

        if validation_split and not validation_data:
            if not (
                isinstance(x, tf.data.Dataset),
                isinstance(x, typing.Generator),
                isinstance(x, tf.keras.utils.Sequence),
            ):
                train_xy, val_xy = train_validation_split(
                    (x, y), validation_split=validation_split
                )
                x, validation_data = (
                    tf.data.Dataset.from_tensor_slices(train_xy),
                    tf.data.Dataset.from_tensor_slices(val_xy),
                )
                if batch_size:
                    x = x.batch(batch_size)
                if validation_batch_size:
                    validation_data = validation_data.batch(validation_batch_size)

        self.validation_data = validation_data
        self.validation_steps = validation_steps

        if self.validation_data is None:  # TODO: Should this check be here or in
            # surface plot???
            raise ValueError("Need validation_data")

        if callbacks:
            # TODO: add overwrite option
            callbacks.append(CheckpointCallback(path=self._get_cpoint_path()))
        else:
            callbacks = [CheckpointCallback(path=self._get_cpoint_path())]

        keys = [
            "y",
            "epochs",
            "verbose",
            "callbacks",
            "validation_split",
            "shuffle",
            "class_weight",
            "sample_weight",
            "initial_epoch",
            "steps_per_epoch",
            "validation_steps",
            "validation_freq",
            "max_queue_size",
            "workers",
            "use_multiprocessing",
        ]
        return self.__getattr__("fit")(
            x=x,
            y=y,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            validation_data=self.validation_data,
        )

    def _new_model(self, run_eagerly=False):
        """ create a new model for evaluation """

        config = self.__getattr__("get_config")()
        if type(self.model) == tf.python.keras.engine.functional.Functional: # need to test type instead of isinstance bc Sequential is Functional
            new_mod = tf.keras.Model.from_config(config)
        else:
            new_mod = tf.keras.Sequential.from_config(config)
        new_mod.set_weights(self.__getattr__("get_weights")())
        new_mod.compile(optimizer=self.opt, loss=self.loss,
                run_eagerly=run_eagerly)

        return new_mod

    def surface_plot(self, objs=None, normalizer_config={'alphas_size':35,
        'betas_size':35, 'extension':1, 'quiet':False}):
        if objs:
            objs = [self, *objs]
        else:
            objs = self

        ct = CalcTrajectory()
        ct.fit(objs)
        surface = normalizer(objs, ct, **normalizer_config)
        fig = plotter.make_figure([plotter.make_trace(dat) for dat in surface.values()])
        return fig 
