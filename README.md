# Learnable FIR Filter-banks Time-Convolutional Layers

[1] Ahmed Imtiaz Humayun, Md. Tauhiduzzaman Khan, Shabnam Ghaffarzadegan, Zhe Feng and Taufiq Hasan, “An Ensemble of Transfer, Semi-supervised and Supervised Learning Methods for Pathological Heart Sound Classification”, Interspeech 2018, Hyderabad, India.

[2] Ahmed Imtiaz Humayun, Shabnam Ghaffarzadegan, Zhe Feng and Taufiq Hasan, “Learning Front-end Filter-bank Parameters using Convolutional Neural Networks for Abnormal Heart Sound Detection”, Proc. EMBC 2018, Hawaii, USA.
Keras Wrappers for Learnable - FIR built on top of Tensorflow backend

### Contents:

- Learnable FIR
- Learnable Linear Phase FIR
- Learnable Zero Phase FIR
- Keras DCT Layer

### Requirements:
- Keras
- Tensorflow

### Usage:
`` from custom_layers import Conv1D_linearphase
``

Use it like any convolutional layer in Keras.


```

from keras.models import Model
from keras.layers import Input, Dense, Conv1D
from custom_layers import Conv1D_linearphase

x = Input(shape=(32,))
x = Conv1D_linearphase(3, 61, padding='valid') # Filterbank of 3 filters, outputs stacked along channel axis
x = Conv1D(8, 5, padding='same', activation='relu')
x = Dense(32)(x)
model = Model(inputs=a, outputs=b)

```
