# cspell: disable
from .Mlag import Mlag
import tensorflow.python as tfp

"""
    Provide support for functional model method calls
"""


class ModelVFunc(Mlag, tfp.keras.engine.functional.Functional):
    def __init__(self, model, msaver_path="vwd", verbose=1):
        """
        Args:
            model: tensorflow.python.keras.engine.functional.Functional instance
            msaver_path: string containing path of directory to save model
            checkpoints in
        """
        Mlag.__init__(self, model, msaver_path, verbose)
        tfp.keras.engine.functional.Functional.__init__(
            self, model.inputs, model.outputs, model.name, model.trainable
        )
