from keras.models import Model, load_model

from load_model import load_model
from load_data import load_training_data, load_development_data
import keras

import numpy as np
import os
from os import path
import time

os.environ["CUDA_VISIBLE_DEVICES"]="1,3"

tic = time.time()


Tx = 10000
n_x = 2482
n = 255

if(path.exists('training_progress.csv')):
	progress = np.loadtxt('training_progress.csv', delimiter=',').tolist()

else:
	progress = []

if(path.exists('models/min_model.h5')):
	model = load_model(Tx, n_x, n, location='models/min_model.h5')

else:
	model = load_model(Tx, n_x, n)
	
model.compile(optimizer='adam', loss='mse')

X_train, Y_train, X_train_gender = load_training_data()
X_dev, Y_dev, X_dev_gender = load_development_data()

toc = time.time()

print("Elapsed time :", toc-tic)

no_of_downward_epochs = 1000

if(path.exists('learner_params.txt')):
	learner_params = np.loadtxt('learner_params.txt')
	min_loss_dev = learner_params[0]
	prev_loss_dev = learner_params[1]
	loss_dev = learner_params[2]

	increase_count = int(learner_params[3])
	current_epoch_number = int(learner_params[4])
	total_epoch_count = int(learner_params[5]) + 1

else:	
	min_loss_dev = 10000
	prev_loss_dev = 10000
	loss_dev = 10000

	increase_count = 0
	current_epoch_number = 1
	total_epoch_count = 1


increase_count_threshold = 10000



while(current_epoch_number < no_of_downward_epochs):
	
	print(str(total_epoch_count)*30)
	print(no_of_downward_epochs - current_epoch_number, "epochs to go.")

	prev_loss_dev = loss_dev
	
	losses = []

	for (X_batch, Y_batch, X_gender_batch) in zip(X_train, Y_train, X_train_gender):
		l = model.fit([X_batch, X_gender_batch], Y_batch, batch_size = X_batch.shape[0])
		losses.append(l.history['loss'][-1])

	loss_train = sum(losses)/len(losses)
	loss_dev = model.evaluate([X_dev, X_dev_gender], Y_dev, batch_size = X_dev.shape[0])

	print(loss_train, loss_dev)

	if(loss_dev < min_loss_dev):
		min_loss_dev = loss_dev
		model.save('models/min_model.h5')
		increase_count = 0
		current_epoch_number = current_epoch_number + 1

	else:

		if(loss_dev >= prev_loss_dev):
			increase_count = increase_count + 1

			if(increase_count >= increase_count_threshold):
				model = keras.models.load_model('models/min_model.h5')
				increase_count = 0
				prev_loss_dev = min_loss_dev
				print("\nGOING BACK TO THE min_model!")

		else:
			increase_count = increase_count - 1

			if(increase_count < 0):
				increase_count = 0

	progress.append([total_epoch_count, loss_train, loss_dev])
	np.savetxt('training_progress.csv', np.array(progress), fmt='%.4f', delimiter=',')
	np.savetxt('learner_params.txt', np.array([min_loss_dev, prev_loss_dev, loss_dev, increase_count, current_epoch_number, total_epoch_count]))

	total_epoch_count = total_epoch_count + 1
	print("\n\n")