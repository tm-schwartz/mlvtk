import tensorflow as tf

"""
    Callback to save model data to disk
"""

# cspell: disable
class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, path, overwrite=True):
        super(CheckpointCallback, self).__init__()
        self.epoch = 0
        self.path = path
        self.overwrite = overwrite

    def on_epoch_end(self, batch, logs=None):
        self.model.save(
            f"{self.path}model_{self.epoch}.h5",
            overwrite=self.overwrite,
            save_format="h5",
        )
        self.epoch += 1
