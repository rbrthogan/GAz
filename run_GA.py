__author__ = 'Robert Hogan'

from polynomial_GA import *
import cProfile

data=np.loadtxt('data/train')
xdata=data[:3000,:5]
ydata=data[:3000,-1]


testGA=polyGA(terms=10,selection='tournament',tournmanent_size=10)
testGA.load_training_data(xdata,ydata)

testGA.run_GA()
testGA.save_poly()
testGA.save_y_prediction()

