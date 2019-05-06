import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, LSTM, Input, Concatenate, Dropout
import keras


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (10000, 12,))
	X_gender = Input(shape = (1,))

	Y = LSTM(9, name = 'lstm_cell')(X)
	Y = Dropout(rate = 2/9)(Y)

	Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(6, activation = 'relu')(Y)
	Y = Dropout(rate = 1/3)(Y)

	Y = Dense(3, activation = 'relu')(Y)
	Y = Dropout(rate = 1/3)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model