__author__ = 'Robert Hogan'
import sys, getopt
import numpy as np
import copy
def genome2poly(individual,x):
    '''
    Makes polynomial with all coefficients set to 1 from genetic encoding and xdata
    '''
    terms=len(individual)
    vars=len(individual[0])
    t=np.empty((terms,len(x),vars))
    #for each term of individual raise data to powers specified

    for i in range(terms):
        t[i,:,:]=x**np.array(individual[i])
    #muliply together subterms e.g. [x1^2,x2^0]->x1^2*x2^0
    p=t.prod(axis=2)
    return p
def y_model(poly):
        '''
        Returns the polynomial model function for an individual

        Args:
            poly: polynomial generated for individual using genome2poly
        '''
        def y_model_i(x,*coeffs):
            #calculated full polynomial using given coefficients
            y=np.dot(coeffs,poly)
            return y
        return y_model_i

def y_deriv(individual,t):
    def y_deriv_i(xdata,*coeffs):
        poly_d=copy.deepcopy(individual)
        poly_d[:,t]-=1
        poly_d[poly_d<0]+=1
        poly_d_coeffs=individual[:,t]*coeffs

        p=genome2poly(poly_d,xdata)

        #muliply by coeffs and sum to get predicted y value for each data point
        y=np.dot(poly_d_coeffs,p)

        return np.squeeze(y)
    return y_deriv_i

def error_bars(p,x,errors):
    individual=p[:,1:]
    vars=len(individual[0])
    coeffs=p[:,0]
    error_sq=0
    for i in range(vars):
        error_sq+=(y_deriv(individual,i)(x,coeffs)*errors[:,i])**2
    return np.sqrt(error_sq)

def main(argv):
    poly_file="poly_out"
    Errors=False
    try:
        opts, args = getopt.getopt(argv,"f:p:o:e:",["file=","polynomial=","outfile=","errors="])
    except getopt.GetoptError:
        print 'option error'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-f':
            data_file=arg
        elif opt == '-p':
            poly_file=arg
        elif opt == '-e':
            if str(arg)=='Y' or str(arg)=='y':
                Errors=True
            elif str(arg)=='N' or str(arg)=='n':
                Errors=False
            else:
                print "Don't understand error option. Not using errors"
        elif opt == '-o':
            outfile=arg
    try:
        data_file
    except:
        print "Please input data file with flag -f"
    try:
        outfile
    except:
        outfile=data_file+"_predictions"

    p=np.loadtxt(poly_file)

    best_individual=p[:,1:]
    best_coeffs=p[:,0]
    vars=len(best_individual[0])
    data=np.loadtxt(data_file)
    x=data[:,:vars]
    poly=genome2poly(best_individual,x)
    y_predict=np.squeeze(y_model(poly)(x,best_coeffs))

    if not Errors:
        np.savetxt(outfile,y_predict)
    else:
        errors=data[:,vars:-1]
        y_err=error_bars(p,x,errors)
        np.savetxt(outfile,np.vstack((y_predict,y_err)).transpose())


if __name__ == "__main__":
    main(sys.argv[1:])