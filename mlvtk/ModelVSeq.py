#cspell: disable
from Mlag import Mlag
import copy
import os
import h5py
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

class ModelVSeq(tf.keras.Sequential, Mlag):
    def __init__(self, model, msaver_path='vwd'):
        super().__init__(model.inputs, model.outputs,
                                               model.name, model.trainable,
                                               model, msaver_path)
