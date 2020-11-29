""" Code to make a model from a dictionary """


import tensorflow as tf
from mlvtk import create_model
import tensorflow.keras as tfkeras
from tensorflow.keras.layers import Dense, BatchNormalization, LayerNormalization, ActivityRegulation
from tensorflow.keras.activations import relu, elu, selu, swish, softmax, softplus, softsign, tanh, hard_sigmoid

def read_dict(params: dict):

    """ `params` should be a dictionary with keys:
            `layers`: list of chars
            `n_neurons`: list of ints
            `optimizer`: optimizer instance or string
            `learning_rate`: float or scheduler
            `norm_reg_kwarg`: list of list of kwargs to pass to BN, LN, or AR
                            layers or None
            `loss`: str, or loss function instance
        

        `layers` should be a list of chars 
        """

    layers_dict = dict(d=Dense, bn=BatchNormalization, ln=LayerNormalization,
            ar=ActivityRegulation) 

    activations_dict = dict(r=relu, e=elu, s=selu, sw=swish, t=tanh, sm=softmax,
            sp=softplus, ss=softsign, hs=hard_sigmoid)

    initial_model = tf.keras.Sequential([
        layers_dict[l](units=n, activation=a) if l == 'd' else (layers_dict[l]() if
        isinstance(params.get('norm_reg_kwarg'), None) or
        isinstance(params.get('norm_reg_kwarg')[0], None) else
        layers_dict[l](*params['norm_reg_kwarg'].pop())) for l, n, a in zip(params['layers'], params['n_neurons'],
        params['activations'])
        ])

    model = create_model(initial_model)

    if isinstance(params['optimizer'], str) and not isinstance(params['learning_rate'], None): 
        opt = tf.keras.optimizers.get(params['optimizer'])

    # TODO: Finish optimizer and loss parts

    model.compile(optimizer=opt(learning_rate=params['learning_rate'])
