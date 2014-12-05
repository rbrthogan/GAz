__author__ = 'Robert Hogan'

from polynomial_GA import *
import cProfile

# Load data from file
data=np.loadtxt('data/train')
x_train=data[:300,:5]
y_train=data[:300,-1]
data=np.loadtxt('data/valid')
x_valid=data[:,:5]
y_valid=data[:,-1]
data=np.loadtxt('data/train')
x_test=data[:,:5]
y_test=data[:,-1]

# initialise polynomial GA object by setting hyperparameters
GA=polyGA(terms=10,selection='tournament',tournmanent_size=10)

#load data into GA object
GA.load_training_data(x_train,y_train)
GA.load_validation_data(x_valid,y_valid)
GA.load_test_data(x_test,y_test)

#run GA
GA.run_GA(max_gens=1)

#save best polynomial
GA.save_poly()
#save prediction for given dataset set
GA.save_y_prediction('test')

