# cspell: disable
import tensorflow as tf
from ModelVSeq import ModelVSeq
from ModelVFunc import ModelVFunc


"""
class seq_factory(tfkeras.Sequential):
    def __init__(self, model):
        super(seq_factory, self).__init__()

        
class func_factory(tf.python.keras.engine.functional.Functional):
    def __init__(self, model):
        super(func_factory, self).__init__(model.inputs, model.outputs)

def conditional_instantiation(model):
    if isinstance(model, tfkeras.Sequential):
        return seq_factory(model)
    return func_factory(model)
"""



def create_model(model):
    if isinstance(model, tf.keras.Sequential):
        return ModelVSeq(model)
    elif isinstance(model, tf.python.keras.engine.functional.Functional):
        return ModelVFunc(model)

