# MLVTK  [![PyPI - Python Version](https://img.shields.io/badge/python-3.6.1%20|%203.7%20|%203.8%20|%203.9-brightgreen)](https://badge.fury.io/py/mlvtk) ![PyPI](https://img.shields.io/pypi/v/mlvtk?color=brightgreen&label=PyPI)
### A loss surface visualization tool


<img alt="Png" src="https://raw.githubusercontent.com/tm-schwartz/mlvtk/dev/visuals/adamax.png" width="80%" />

_Simple DNN trained on MNIST data set, using Adamax optimizer_

---

<img alt="Gif" src="https://raw.githubusercontent.com/tm-schwartz/mlvtk/dev/visuals/gifs/sgd3.gif" width="80%" />

_Simple DNN trained on MNIST, using SGD optimizer_

---

<img alt="Gif" src="https://raw.githubusercontent.com/tm-schwartz/mlvtk/dev/visuals/gifs/adam2.gif" width="80%" />

_Simple DNN trained on MNIST, using Adam optimizer_

---

<img alt="Gif" src="https://raw.githubusercontent.com/tm-schwartz/mlvtk/dev/visuals/gifs/sgd1.gif" width="80%" />

_Simple DNN trained on MNIST, using SGD optimizer_




## Why?

- :shipit: **Simple**: A single line addition is all that is needed.
- :question: **Informative**: Gain insight into what your model is seeing.
- :notebook: **Educational**: *See* how your hyper parameters and architecture impact your
  models perception.


## Quick Start

Requires | version
-------- | -------
python | >= 3.6.1 
tensorflow | >= 2.3.1, <  2.4.2
plotly | >=4.9.0

Install locally (Also works in google Colab!):
```sh
pip install mlvtk
```

Optionally for use with jupyter notebook/lab:

*Notebook*
---
```sh
pip install "notebook>=5.3" "ipywidgets==7.5"
```

*Lab*
---
```sh
pip install jupyterlab "ipywidgets==7.5"

# Basic JupyterLab renderer support
jupyter labextension install jupyterlab-plotly@4.10.0

# OPTIONAL: Jupyter widgets extension for FigureWidget support
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.10.0
```

### Basic Example

```python
from mlvtk.base import Vmodel
import tensorflow as tf
import numpy as np

# NN with 1 hidden layer
inputs = tf.keras.layers.Input(shape=(None,100))
dense_1 = tf.keras.layers.Dense(50, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(10, activation='softmax')(dense_1)
_model = tf.keras.Model(inputs, outputs)

# Wrap with Vmodel
model = Vmodel(_model)
model.compile(optimizer=tf.keras.optimizers.SGD(),
loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# All tf.keras.(Model/Sequential/Functional) methods/properties are accessible
# from Vmodel

model.summary()
model.get_config()
model.get_weights()
model.layers

# Create random example data
x = np.random.rand(3, 10, 100)
y = np.random.randint(9, size=(3, 10, 10))
xval = np.random.rand(1, 10, 100)
yval = np.random.randint(9, size=(1,10,10))

# Only difference, model.fit requires validation_data (tf.data.Dataset, or
# other container
history = model.fit(x, y, validation_data=(xval, yval), epochs=10, verbose=0)

# Calling model.surface_plot() returns a plotly.graph_objs.Figure
# model.surface_plot() will attempt to display the figure inline

fig = model.surface_plot()

# fig can save an interactive plot to an html file,
fig.write_html("surface_plot.html")

# or display the plot in jupyter notebook/lab or other compatible tool.
fig.show()
```
