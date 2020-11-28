# cspell: disable
from .Mlag import Mlag
import tensorflow as tf

"""
    Provide support for sequential model method calls
"""


class ModelVSeq(Mlag, tf.keras.Sequential):
    def __init__(self, model, msaver_path="vwd", verbose=1):
        """
        Args:
            model: tensorflow.keras.Sequential instance
            msaver_path: string containing path of directory to save model
            checkpoints in
        """
        Mlag.__init__(self, model, msaver_path, verbose)
        tf.keras.Sequential.__init__(self, model.layers, model.name)
