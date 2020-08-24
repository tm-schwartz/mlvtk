# cspell: disable
import tensorflow as tf
from ModelVSequential import ModelVSequential
from ModelVFunctional import ModelVFunctional 


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



def modelv(model):
    if isinstance(model, tf.keras.Sequential):
        return ModelVSequential(model)
    #elif isinstance(model, tf.python.keras.engine.functional.Functional):
    return ModelVFunctional(model)

