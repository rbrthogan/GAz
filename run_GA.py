__author__ = 'Robert Hogan'

'''
Main script for running polynomial GA
'''
from polynomial_GA import *

# Load data from file
data=np.loadtxt('data/train')
x_train=data[:,:5]
y_train=data[:,-1]
data=np.loadtxt('data/valid')
x_valid=data[:,:5]
y_valid=data[:,-1]
data=np.loadtxt('data/test')
x_test=data[:,:5]
y_test=data[:,-1]

# initialise polynomial GA object by setting hyperparameters
# see polynomial_GA.py for full list and description of inputs
GA=polyGA(selection='tournament',tournmanent_size=10)

#load data into GA object
GA.load_training_data(x_train,y_train)
GA.load_validation_data(x_valid,y_valid)
GA.load_test_data(x_test,y_test)

#run GA
GA.run_GA()

#save best polynomial
GA.save_poly()
#save prediction for given dataset set
errors=x_test=data[:,5:-1]
GA.save_y_prediction('test',errors=errors)

