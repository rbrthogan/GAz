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


Using optimal polynomial on new dataset:

-Run make_predictions.py script with new data as input:

    options:
 
        -f : path to input data (required)
 
        -e : use input errors to calculate error on prediction (optional, default won't use errors)
            inputs: Y/y (use errors) N/n (don't use errors)
      
        -o : path to save output prediction (optional, defaults to "path/to/input/_predictions")
 
        -p : path to polynomial file ouput by code (optional, defaults to 'poly_out')
 
-----------------------------------------------------------------------------------------

DEPENDENCIES:

- Scipy
- Numpy
- Matplotlib (if using plotting scripts)