# MLVTK
### A loss surface visualization tool


<img alt="Png" src="https://raw.githubusercontent.com/tm-schwartz/mlvtk/dev/visuals/adamstatic.png" width="60%" />

_Simple DNN trained on Wine data set, using Adam optimizer_


## Why?

- :shipit: **Simple**: A single line addition is all thats needed.
- :question: **Informative**: Gain insight into what your model is seeing.
- :notebook: **Educational**: *See* how your hyperparameters and architecture impact your
  models perception.


## Quick Start

Requires | version
-------- | -------
python | >= 3.6.0 
tensorflow | 2.3.x
plotly | 4.9.0

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

### Usage

```python
# construct standard 3 layer network
inputs = tf.keras.layers.Input(shape=(None,784))
dense_1 = tf.keras.layers.Dense(50, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(np.unique(label_train, axis=0).size, activation='softmax')(dense_1) # hard coded outputs size
_model = tf.keras.Model(inputs, outputs)

# create mlvtk model
model = create_model(_model)

# compile and fit like a standard tensorflow model
model.compile(optimizer=tf.keras.optimizers.SGD(),
loss=tf.keras.losses.CategoricCategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=0)

# add title to surface plot
model.surface_plot(title_text=f'Data: {dataname}, Epochs: {epochs}, Optimizer: {model.opt}, LR: {lr}')

model.interp_plot(title=f'Data: {dataname}, Epochs: {epochs}, Optimizer: {model.opt}, LR: {lr}')

```
