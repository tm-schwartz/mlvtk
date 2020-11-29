# cspell: disable
import tensorflow as tf
import tensorflow.python as tfp

from .ModelVFunc import ModelVFunc
from .ModelVSeq import ModelVSeq


def create_model(model):
    """
    Instantiate the corresponding class depending on model type.

    Args:
        model: a tensorflow.python.keras.engine.functional.Functional or
        tensorflow.keras.Sequential model

    Returns:
        ModelVSeq or ModelVFunc

    Raises:
        NotImplementedError: if non-sequential and non-functional model instance
        is passed.
    """
    if isinstance(model, tf.keras.Sequential):
        return ModelVSeq(model)
    elif isinstance(model, tfp.keras.engine.functional.Functional):
        return ModelVFunc(model)
    else:
        raise NotImplementedError("Only Sequential and Functional models are supported")
