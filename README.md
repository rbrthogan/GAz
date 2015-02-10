GAz
===

A genetic algorithm that finds optimal polynomial form to fit to data.

This code was created and optimised for calculating photometric redshifts for low redshift galaxies.

A full description of the algorithm (an its application to photometric redshift estimation) can be found in the preprint http://arxiv.org/abs/1412.5997. Please acknowledge this work when using GAz.


-----------------------------------------------------------------------------------------

USAGE:

-Specify path to training, validation, and test sets in run_GA.py (defaults to use sample data provided)

-Specify GA properties desired (currently optimised for task described in http://arxiv.org/abs/1412.5997) by modifying the polyGA object in run_GA.py (described in polynomial_GA.py)

-Run the run_GA.py script.

NOTE: it is a good idea to run the GA several times to mitigate against the dependence on the intial population. The final result should be taken as the run with the losest cross validation error.

-----------------------------------------------------------------------------------------

DEPENDENCIES:

- Scipy
- Numpy
- Matplotlib (if using plotting scripts)