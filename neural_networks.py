# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:15:41 2019

This file handles everything about neural networks

"""

from tensorflow import keras as ks
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
import keras.backend as B
from sklearn.preprocessing import StandardScaler

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   CONSTANTS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# DEFAULT PARAMETERS
TRAIN_FROM_EXISTING = False

FEATURE_NB = 9
CLASS_NB = 3
EPOCHS_NB = 3
BATCH_SIZE = 256
TEST_SIZE = 0.3

LAYERS_SIZES = [FEATURE_NB, 66, 22, CLASS_NB]
LAYERS_ACTIVATIONS = ['relu', 'relu', 'tanh', 'softmax']

LOAD_MODEL_PATH = '../Data/models/MAVEN_mlp_V1.h5'
#SAVE_MODEL_PATH = '../Data/models/last_model.h5'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   FUNCTIONS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
Function training a neural network according to some parameters and dataset

Inputs:
    pandas.DataFrame()[] List of: X_train, X_test, y_train, y_test (preprocessed)
    int[]                List of layers sizes
    activation[]         List of layers activation
    int                  Number of epoch
    int                  Batch size
    float                Test proportion (between 0 and 1)
"""
def run_training(datasets, layers_sizes = LAYERS_SIZES, layers_activations = LAYERS_ACTIVATIONS, epochs_nb = EPOCHS_NB,
                 batch_size = BATCH_SIZE, test_size = TEST_SIZE, dropout = 0.0):
    if TRAIN_FROM_EXISTING:
        ANN = load_model(LOAD_MODEL_PATH)
    else:
        ANN = create_model(layers_sizes, layers_activations, dropout = dropout)
    training = compile_and_fit(ANN, datasets[0], datasets[2], epochs_nb, batch_size)
    return ANN, training

"""
Creates a keras model
Arguments:
    - lay_s : ex : layer_sizes = [11,11,6] will give an ANN with layers (11, 11, 6)
    - act : ex : act = ['relu', 'relu', 'softmax']
    - dropout : dropout proportion, default to 0 (applied between every layer)
"""
def create_model(lay_s, act, dropout=0.0):
    #initializing the model
    model = Sequential()
    #adding the input layer
    model.add(Dense(lay_s[0], activation = act[0], input_shape=(lay_s[0],)))
    if dropout > 0:
        model.add(Dropout(dropout))
    #adding the other layers
    for i in range(1,len(lay_s)):
        model.add(Dense(lay_s[i], activation = act[i]))
        if dropout > 0:
            model.add(Dropout(dropout))
    return model

"""
Defines the jaccard distance as a custom metrics for keras
CODE COPIE COLLE DEPUIS https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
"""
"""
Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
This loss is useful when you have unbalanced numbers of pixels within an image
because it gives all classes equal weight. However, it is not the defacto
standard for image segmentation.
For example, assume you are trying to predict if each pixel is cat, dog, or background.
You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
The loss has been modified to have a smooth gradient as it converges on zero.
This has been shifted so it converges on 0 and is smoothed to avoid exploding
or disappearing gradient.
Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
= sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
# References
Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
What is a good evaluation measure for semantic segmentation?.
IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
https://en.wikipedia.org/wiki/Jaccard_index
"""

"""
Explication du Jaccard
Les ensembles X et Y à considérer sont:
    Xc : éléments de classe c dans les prédictions
    Yc : éléments de classe c dans le set de test

Pour la précision, les ensembles considérés sont différents:
    Xc : éléments de classe c dans les prédictions
    Y : ensemble des éléments du set de test

Dans les 2 cas on somme ensuite sur l'ensemble des classes c

A noter que le jaccard et la précision sont donc identiques dans le cas d'une classification binaire
"""
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = B.sum(B.abs(y_true * y_pred), axis=-1)
    sum_ = B.sum(B.abs(y_true) + B.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

"""
Compiling and fitting the model

We will now compile the model and train it to fit our training data. This happens
thanks to the Keras .compile() and .fit() methods. The code is as follows:

    #for classification problems, the metrics used will be metrics=['accurracy']
    model.compile(optimizer=, loss=, metrics=)
    model.fit(X_train, y_train, epochs=, batch_size=, verbose=)

The fit functions returns the history, which can be saved as a variable and used
for results visualization.

Keras offers a lot of different possible loss functions. A few examples:
    - mean_squared_error
    - categorical_crossentropy
    - poisson
    - hinge
    - cosine_proximity
    ...

Same for optimizers:
    - SGD
    - RMSprop
    - Adagrad
    - Adam
    ...

The batch_size parameter defines how many samples will be averaged before tuning
the parameters.
the epochs parameter sets the number of consecutive trainings.



CALLBACK functions:
    Keras enables the user to use callback arguments, which allow to take a look
    at the state of the network during the training. Callbacks can be passed to
    the model as follows :
        model.fit(X_train, y_train, epochs=, batch_size=, verbose=, callbacks=)
"""

"""
Arguments:
    - model : model to compile and train
    - X_train : train set to feed to the network
    - y_train : labels corresponding to the X_train data set
    - n_epochs : number of epochs of training
    - b_s : batch size during the training
    - loss_name : loss to use (default: jaccard_distance)
Returns:
    training is the history of the training
"""
def compile_and_fit(model, X_train, y_train, n_epochs, b_s, val_size=0, loss_name = jaccard_distance):
    #optimizer = optimizers.SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer = optimizer, loss = [loss_name], metrics = ['acc'])
    model.compile(optimizer = 'adam', loss = [loss_name], metrics = ['acc'])
    training = model.fit(X_train, y_train, validation_split = val_size, epochs = n_epochs, batch_size = b_s, verbose = 1)
    return training

"""
Get a timed vector of predictions from the test set
"""
def get_pred_timed(model, X_test_timed, scale_data_timed):
    y_pred_timed = pd.DataFrame()
    y_pred_timed['epoch'] = X_test_timed['epoch']

    scale_data = scale_data_timed.copy()
    del scale_data['epoch']

    X_test = X_test_timed.copy()
    del X_test['epoch']

    scaler = StandardScaler().fit(scale_data)
    X_test = scaler.transform(X_test)

    y_pred = model.predict_classes(X_test)
    y_pred_timed['label'] = y_pred

    return y_pred_timed

"""
Get the probability with which the model predicted each class
"""
def get_prob_timed(model, X_test_timed, X_train_timed):
    y_prob_timed = pd.DataFrame()
    y_prob_timed['epoch'] = X_test_timed['epoch']

    X_train = X_train_timed.copy()
    del X_train['epoch']

    X_test = X_test_timed.copy()
    del X_test['epoch']

    scaler = StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)

    y_prob = model.predict(X_test)
#    y_prob_timed['prob'] = [max(y_prob[i]) for i in range(X_test.shape[0])]
    y_prob_timed['prob_ev'] = [y_prob[i][0] for i in range(X_test.shape[0])]
    y_prob_timed['prob_sh'] = [y_prob[i][1] for i in range(X_test.shape[0])]
    y_prob_timed['prob_sw'] = [y_prob[i][2] for i in range(X_test.shape[0])]

    return y_prob_timed

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            UTILITY FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
Saving the model to a dedicated file
"""
def save_model(filepath, model):
    model.save(filepath)

"""
Loading the model from a specific file
"""
def load_model(filepath = LOAD_MODEL_PATH):
    model = ks.models.load_model(filepath, custom_objects={'jaccard_distance': jaccard_distance})
    return model

if __name__ == '__main__':
    print('mlp main')