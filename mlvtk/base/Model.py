#TODO replace calls to self.checkpoint_path with self._get_cpoint_path
#TODO allow for creation of model w/o precalling tf...Model/Sequential
import pathlib
import shutil
import typing
from typing import Union

import tensorflow as tf
from tensorflow.python.keras.engine.data_adapter import train_validation_split
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.keras import Sequential, Model

from .callbacks.CheckpointCallback import CheckpointCallback


class Model:
    def __init__(self, model:Union[Functional,Sequential, Model],
            checkpoint_path:Union[pathlib.Path, str] ="vwd", verbose:int =0): #TODO add
                                                                 # `inputs`/`outputs` to enable functional model creation

        self.model = model
        self.verbose = verbose
        self.overwrite = False
        self.checkpoint_path:typing.Union[pathlib.Path, str] = pathlib.Path(checkpoint_path)

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

        return self.__getattr__('compile')(*args, **kwargs)


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

        self.checkpoint_path = pathlib.Path(self.checkpoint_path)

        if (
            self.checkpoint_path.exists()
            and self.checkpoint_path.is_dir()
            and self.checkpoint_path.lstat().st_size
        ):

            if self.overwrite is not None:
                if self.overwrite:
                    shutil.rmtree(self.checkpoint_path)
                elif self.overwrite == False:
                    self.checkpoint_path = input("Please enter new save folder path ")
                    self.checkpoint_path = pathlib.Path(self.checkpoint_path)

            else:
                overwrite = input(
                    f"{self.checkpoint_path} is not empty, overwrite? (this message can be silenced by setting `overwrite=True/False`)"
                )
                while True:
                    if overwrite in ["no", "n"]:
                        self.checkpoint_path = input(
                            "Please enter new save folder path "
                        )
                        self.checkpoint_path = pathlib.Path(self.checkpoint_path)
                        break
                    elif overwrite in ["yes", "y"]:
                        shutil.rmtree(self.checkpoint_path)
                        break
                    overwrite = input("please enter: yes/no/y/n")

        if not self.checkpoint_path.is_dir():
            self.checkpoint_path.mkdir(parents=True)

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
            callbacks.append(CheckpointCallback(path=self.checkpoint_path))
        else:
            callbacks = [CheckpointCallback(path=self.checkpoint_path)]

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
        return self.__getattr__('fit')(x=x, y=y, epochs=epochs, verbose=verbose,
callbacks=callbacks, validation_split=validation_split, shuffle=shuffle,
class_weight=class_weight, sample_weight=sample_weight,
initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
validation_steps=validation_steps, validation_freq=validation_freq,
max_queue_size=max_queue_size, workers=workers,
use_multiprocessing=use_multiprocessing, validation_data=self.validation_data)

    def _new_model(self):
        """ create a new model for evaluation """

        config = self.__getattr__('get_config')()
        if isinstance(self.model, Functional):
            new_mod = tf.keras.Model.from_config(config)
        else:
            new_mod = tf.keras.Sequential.from_config(config)
        new_mod.set_weights(self.__getattr__('get_weights')())
        new_mod.compile(optimizer=self.opt, loss=self.loss, run_eagerly=False)

        return new_mod
