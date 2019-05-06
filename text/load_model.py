import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Bidirectional, TimeDistributed, Lambda, Flatten, Activation, Multiply
import keras
import keras.backend as K

def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))
	X_gender = Input(shape = (1,))

	Y = CuDNNLSTM(200, name = 'lstm_cell', return_sequences = True)(X)
	#print(Y.shape)
	P = TimeDistributed(Dense(40, activation = 'tanh'))(Y)
	alpha = TimeDistributed(Dense(1, activation = None))(P)

	alpha = Lambda(lambda x: K.squeeze(alpha, axis = -1))(alpha)
	alpha = Activation('softmax')(alpha)
	alpha = Lambda(lambda x: K.expand_dims(alpha, axis = -1))(alpha)

	F = Multiply()([alpha, Y])
	F = Lambda(lambda x: K.sum(F, axis = 1))(F)
	F = Lambda(lambda x: F*100)(F)
	
	Y = Dropout(rate = 0.3)(F)

	Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(60, activation = 'relu')(Y)
	Y = Dropout(rate = 0.3)(Y)
	
	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model


if __name__ == "__main__":
	m = load_model()