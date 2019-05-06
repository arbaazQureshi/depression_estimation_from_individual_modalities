import numpy as np
import pandas as pd




def load_training_data():


	X_location = '/home/syedcs15/depression_estimation/preprocessed_data/training_data/speech/preprocessed_COVAREP_output/training_COVAREP_output.npy'
	X = np.load(X_location)
	

	Y_location = '/home/syedcs15/depression_estimation/preprocessed_data/training_data/speech/preprocessed_COVAREP_output/training_PHQ_scores.npy'
	Y = np.load(Y_location)

	X_gender_location = '/home/syedcs15/depression_estimation/preprocessed_data/training_data/speech/preprocessed_COVAREP_output/training_participant_genders.npy'
	X_gender = np.load(X_gender_location)

	print("Training data is loaded.")

	return (X, Y, X_gender)





def load_development_data():

	X_location = '/home/syedcs15/depression_estimation/preprocessed_data/development_data/speech/preprocessed_COVAREP_output/development_COVAREP_output.npy'
	X = np.load(X_location)

	Y_location = '/home/syedcs15/depression_estimation/preprocessed_data/development_data/speech/preprocessed_COVAREP_output/development_PHQ_scores.npy'
	Y = np.load(Y_location)

	X_gender_location = '/home/syedcs15/depression_estimation/preprocessed_data/development_data/speech/preprocessed_COVAREP_output/development_participant_genders.npy'
	X_gender = np.load(X_gender_location)

	print("Development data is loaded")

	return (X, Y, X_gender)


if(__name__ == "__main__"):
	train = load_training_data()
	dev = load_development_data()