# cspell: disable
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

class ModelVFunctional(tf.python.keras.engine.functional.Functional):
    def __init__(self, model, msaver_path='mlvtk'):
        super(ModelVFunctional, self).__init__(model.inputs, model.outputs,
                model.name, model.trainable)
        self.alphas = np.linspace(-3, 3, 35)
        self.betas = np.linspace(-3, 3, 35)
        self.testdat = None
        self.loss_df = None
        self.xdir = None
        self.ydir = None
        self.evr = None
        self.msave_path = msaver_path
        self._compatible = False
        self._fit = model.fit
        
    def _tf_compatible(self):
        for f in os.listdir(self.msave_path):
            hf = h5py.File(pathlib.Path(self.msave_path, f), 'r+')
            hf.attrs.modify('model_config', hf.attrs.get('model_config').replace(b'"class_name": "ModelVFunctional"', b'"class_name" :"Functional"'))
            hf.close()
        self._compatible = True
        
    def fit(self, *args, **kwargs):
        if kwargs.get('callbacks'):
            kwargs['callbacks'].append()
        return self._fit(*args, )
        
    def _calculate_loss(self):
        # TODO: assert self.alphass.shape == beta.shape in outer scope

        weights = np.asarray(self.get_weights())

        def filter_normalize():
            delta = np.array([np.reshape(np.random.randn(ww.size), ww.shape)
                              for ww in weights], dtype='object')
            etta = np.array([np.reshape(np.random.randn(ww.size), ww.shape)
                             for ww in weights], dtype='object')
            if (bn:= filter(lambda layer: layer.name == 'batch_normalization',
                             self.layers)):
                for layer in bn:
                    i = self.layers.index(layer)
                    delta[i] = np.zeros((delta[i].shape))
                    etta[i] = np.zerose((etta[i].shape))
            paramlist = itertools.product(self.alphas, self.betas)

            def normalize_filter(fw):
                f, w = fw
                return f * \
                    np.array([(np.linalg.norm(w) / (np.linalg.norm(f) + 1e-10))])

            result = np.array([np.array([d0 * a + d1 * b for d0, d1 in
                                         zip(map(normalize_filter, zip(delta, weights)),
                                             map(normalize_filter, zip(etta,
                                                                       weights)))], dtype='object')
                               for a, b in paramlist], dtype='object')

            return result.reshape(
                self.alphas.shape[0],
                self.betas.shape[0],
                len(weights))

        def gen_filter_final():
            for ix, iy in itertools.product(range(self.self.alphas.shape[0],
                                                  range(self.betas.shape[0]))):
                yield (ix, iy, filters[ix, iy])

        def _calc_weights(data):
            return (data[0], data[1], np.array([mw + data[2][k]
                                                for k, mw in enumerate(weights)], dtype='object'))

        def _calc_loss(w):
            with tf.device('/GPU:0'):
                new_mod.set_weights(w)
                new_mod.compile(optimizer=self.optimizer, loss=self.loss)
                return new_mod.evaluate(self.testdat, use_multiprocessing=True,
                                        verbose=0)

        new_mod = tf.keras.models.model_from_config(self.get_config())

        filters = filter_normalize(weights, self.alphas, self.beta)

        df = pd.DataFrame(index=self.alphas, columns=self.betas)

        prog_bar = tf.keras.utils.ProgBar(
            self.alphas.size *
            self.betas.size,
            width=30,
            verbose=1,
            interval=0.05,
            stateful_metrics=None,
            unit_name='step')

        for step, new_weights in enumerate(
                map(_calc_weights, gen_filter_final())):
            df.iloc[new_weights[0], new_weights[1]
                    ] = _calc_loss(new_weights[2])
            prog_bar.update(step + 1)
        self.loss_df = df

        
    def gen_path(self):

        assert os.path.isdir(self.msave_path), 'Could not find model save path. Check "msave_path" is set correctly'
        
        if not self._compatible:
            self._tf_compatible()
        files = [pathlib.Path(self.msave_path, file_name) for file_name in
                 sorted(os.listdir(self.msave_path), key=lambda x:
                        int(x.split('_')[-1][:-3]))]
        final_model = tf.keras.models.load_model(files[-1])
        theta_n = final_model.get_weights()
        raw_weights = [tf.keras.models.load_model(model_file).get_weights() for
                       model_file in files[:-1]]
        weight_differences_btw_epochs = [[theta_i - theta_n_i for theta_i, theta_n_i
                                          in zip(theta, theta_n)] for theta in raw_weights]

        # TODO: rename
        def tensorlist_to_tensor_tf(weights):
            return np.concatenate([w.flatten() if w.ndim > 1 else w for w in
                                    weights])

        def npvec_to_tensorlist_tf(direction, params):
            w2 = copy.deepcopy(params)
            idx = 0
            for i, w in enumerate(w2):

                w2[i] = direction[idx:idx + w.size]
                w2[i] = w2[i].flatten()
                idx += w.size
            return np.concatenate(w2)

        def project_1d_tf(w, d):

            assert len(w) == len(d), 'dimension does not match for w and d'
            return np.dot(np.array(w), d) / np.linalg.norm(d)

        def project_2d_tf(d, dx, dy):

            x = project_1d_tf(d, dx)
            y = project_1d_tf(d, dy)
            return x, y

        pca = PCA(n_components=2)
        T0 = [tensorlist_to_tensor_tf(i)
              for i in weight_differences_btw_epochs]
        pca.fit(np.array(T0))
        pca_1 = pca.components_[0]
        pca_2 = pca.components_[1]

        xdir, ydir = [], []
        for ep in T0:
            xd, yd = project_2d_tf(
                ep, npvec_to_tensorlist_tf(
                    pca_1, theta_n), npvec_to_tensorlist_tf(
                    pca_2, theta_n))
            xdir.append(xd)
            ydir.append(yd)

        self.xdir = xdir
        self.ydir = ydir
        self.evr = pca.explained_variance_ratio_

    def plot(self, title_text=None, save_file=None):

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'is_3d': True}, {'type': 'scatter'}]],
            subplot_titles=['3d surface plot with optimizer path', '1d scatter\
                plot of loss values'],)
        vals = self.loss_df.values.ravel()
        xs = [self.loss_df.index[i]
              for i in np.digitize(self.xdir, self.loss_df.index, right=True)]
        ys = [self.loss_df.columns[i]
              for i in np.digitize(self.ydir, self.loss_df.columns, right=True)]
        zs = [self.loss_df.loc[x, y] for x, y in zip(xs, ys)]

        fig.add_trace(
            go.Surface(
                x=self.loss_df.index,
                y=self.loss_df.columns,
                z=self.loss_df.values,
                opacity=.9,
                showscale=True,
                lighting=dict(
                    ambient=0.6,
                    roughness=0.9,
                    diffuse=0.5,
                    fresnel=2),
                colorscale='haline_r',
                colorbar=dict(
                    lenmode='pixels',
                    len=400)),
            row=1,
            col=1)
        fig.add_scatter3d(
            x=self.xdir,
            y=self.ydir,
            z=zs,
            marker=dict(
                size=2,
                color='red'),
            line=dict(
                color='darkblue',
                width=2),
            showlegend=True,
            name='opt path',
            row=1,
            col=1)

        fig.add_trace(
            go.Scattergl(
                x=np.arange(
                    0,
                    len(vals)),
                y=vals,
                mode='markers',
                showlegend=False),
            row=1,
            col=2)
        
        if title_text:
                 title = dict(text=title_text, x=.7)
        else:
                 title = None
        fig.update_layout(title=None, autosize=False,
                          width=1200, height=900, margin=dict(l=10), bargap=.2,
                          paper_bgcolor="LightSteelBlue")
        fig.show()
        if save_file is not None:
            fig.write_html(f"{save_file}.html")
      