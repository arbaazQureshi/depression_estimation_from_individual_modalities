from load_model import load_model
from keras.models import Model
from load_data import load_training_data, load_development_data

import numpy as np


model = load_model()
model.load_weights('optimal_weights.h5')

new_model = Model(inputs = model.inputs, outputs = model.layers[-5].output)

X_train, y_train, X_train_gender = load_training_data()
X_dev, Y_dev, X_dev_gender = load_development_data()


Y_train_pred = new_model.predict([X_train, X_train_gender])
Y_dev_pred = new_model.predict([X_dev, X_dev_gender])

np.save('transcript_training.npy', Y_train_pred)
np.save('transcript_development.npy', Y_dev_pred)