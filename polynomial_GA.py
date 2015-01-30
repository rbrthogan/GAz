__author__ = 'Robert Hogan'

'''
Main source code for polynomial genetic algorithm.
'''

from hash_table import *
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy

class polyGA:
    def __init__(self,pop_size=100, mutate_prob=0.03,recomb_prob=0.5,\
                 poly_degree=5,terms=20,vars=5, selection='roulette',tournmanent_size=None,verbose=True):
        '''
        Initialise polyGA object

        Args:
            pop_size: population size for genetic algorithm (int, positive,should be even)
            mutate_prob: mutation probability (float, in range (0,1))
            recomb_prob: recombination probability i.e. prob that breeding takes place (float, in range (0,1))
            poly_degree: maximum allowable degree of polynomial (int, positive)
            terms: number of terms in polynomial (int, positive)
            vars: number of variables/features in problem (int, positive)
            selection: selection method (roulette wheel or tournament)
            tournament_size: size of selection tournaments if tournaments used
            verbose: print fitness throughout

        Returns:
            polyGA object
        '''


        if selection=='tournament' and not tournmanent_size:
            raise ValueError('Please input desired tournament size when using tournament selection')
            return
        if selection not in['tournament','roulette']:
            raise ValueError('Please choose valid selection method')
            return
        if pop_size%2 !=0:
            pop_size=pop_size+1
            raise Warning('Population must be even for breeding so it has been increased by one.')
        if mutate_prob <0 or mutate_prob >1 :
            raise ValueError('Please provide valid mutation probability in range [0,1]')
        if recomb_prob <0 or recomb_prob >1 :
            raise ValueError('Please provide valid recombination probability in range [0,1]')

        self.select=selection               #set selection method
        self.tourney_size=tournmanent_size  #set tournament size
        self.pop_size=pop_size              #set size of population
        self.mutate_prob=mutate_prob        #set mutation probability
        self.recomb_prob=recomb_prob        #set recombination probability
        self.degree=poly_degree             #set maximum degree of polynomial
        self.terms=terms                    #set number of terms in polynomial
        self.vars=vars                      #set number of variables in polynomial
        self.verbose=verbose                #print out rms error as code evolves
        return

    def load_training_data(self,xdata,ydata):
        '''
        Load training data into object

        Args:
            xdata: numpy array with shape (# training examples, # features/variables)
            ydata: numpy array with shape (# training examples, )
        '''
        if np.shape(xdata)[1] !=self.vars:
            raise ValueError('Shape of x-data %s does not match that '\
                             'expected for number of features/variables= %s' % (xdata.shape,self.vars))
        self.x=xdata
        self.y=ydata
        return

    def load_validation_data(self,xdata,ydata):
        '''
        Load cross validation data into object

        Args:
            xdata: numpy array with shape (# cv examples, # features/variables)
            ydata: numpy array with shape (# cv examples, )
        '''
        if np.shape(xdata)[1] !=self.vars:
            raise ValueError('Shape of x-data %s does not match that' \
                             'expected for number of features/variables= %s' % (xdata.shape,self.vars))
        self.x_v=xdata
        self.y_v=ydata
        return

    def load_test_data(self,xdata,ydata):
        '''
        Load test data into object

        Args:
            xdata: numpy array with shape (# test examples, # features/variables)
            ydata: numpy array with shape (# test examples, )
        '''
        if np.shape(xdata)[1] !=self.vars:
            raise ValueError('Shape of x-data %s does not match that'\
                             ' expected for number of features/variables= %s' % (xdata.shape,self.vars))
        self.x_t=xdata
        self.y_t=ydata
        return

    def rand_term(self):
        '''
        Generate random term for given polynomial restrictions
        '''
        term=[]
        total=0

        #to make it more likely to generate lower degree terms, first pick max degree of term
        max_power=np.random.randint(0,self.degree+1)

        for i in range(0,self.vars):
            if total==max_power:
                term.append(0)
                continue
            n=np.random.randint(0,max_power-total+1) #generate random int for term i
            total=total+n
            term.append(n)

        np.random.shuffle(term) #shuffle term to rebalance powers over all variables

        return term

    def generate_pop(self):
        '''
        Generate initial population from scratch
        '''
        pop=[] #initialise list to store inidividuals
        for i in range(0,self.pop_size):
            individual=[] #initialise list to store terms of individual
            individual.append([0]*self.vars)
            for j in range(1,self.terms):
                while True:
                    term=self.rand_term()
                    if term not in individual:
                        break
                individual.append(term) #fill individual with distinct but random terms
            individual.sort() #sort individual to ensure encoding is unique (needed for searching stored solutions)
            pop.append(individual)
        self.pop=pop
        return

    def genome2poly(self,individual,x):
        '''
        Makes polynomial with all coefficients set to 1 from genetic encoding and xdata
        '''

        #for each term of individual raise data to powers specified
        t=np.array([x**np.array(individual[i]) for i in range(len(individual))])
        #muliply together subterms e.g. [x1^2,x2^0]->x1^2*x2^0
        p=t.prod(axis=2)
        return p

    def y_model(self,poly):
        '''
        Returns the polynomial model function for an individual

        Args:
            poly: polynomial generated for individual using genome2poly
        '''
        x=self.x
        def y_model_i(x,*coeffs):
            #calculated full polynomial using given coefficients
            y=np.dot(coeffs,poly)
            return y
        return y_model_i

    def jacobian(self,poly):
        '''
        Derivative of model wrt coeffs for an individual. Used for optimisation.
        '''
        def jacobian_i(*coeffs):
            return np.array(poly)
        return jacobian_i

    def optimize_individual(self,individual):
        '''
        Optimise the coefficients for a particular polynomial specified by and individual

        Returns:
            coeffs: optimised coefficients
            fit: fitness of individual
        '''

        #initialise coefficients randomly in range [-1,1]
        init_coeffs=2*np.random.rand(len(individual))-1.0

        poly=self.genome2poly(individual,self.x)
        coeffs=curve_fit(self.y_model(poly),self.x,self.y,p0=init_coeffs,Dfun=self.jacobian(poly),col_deriv=1)[0]

        #Fitness function
        residuals=np.squeeze(self.y_model(poly)(self.x,coeffs))-self.y
        err=np.dot(residuals,residuals)
        fit=1.0/err

        return coeffs,fit

    def get_fitness(self):
        '''
        Tests fitness of all new inidividuals by optimising the coefficients on data.
        If individual has already been seen then result is extracting from stored solution index.
        If there is a new best fitness then the stored best individual is replaced
        '''
        fit=[]
        coeffs=[]
        for individual in self.pop:

            out=self.sol_index.retrieve(individual)

            if out:
                fit_i,coeffs_i=out
                fit.append(fit_i)
                coeffs.append(coeffs_i)
            else:
                coeffs_i,fit_i=self.optimize_individual(individual)
                fit.append(fit_i)
                coeffs.append(coeffs_i)
                self.sol_index.insert([individual,fit_i,coeffs_i])

        self.pop_fit=fit

        #check if there is a new best polynomial, if so update the stored bests
        i=np.argmax(fit)
        if fit[i]>self.best_fit:
            self.best_fit=fit[i]
            self.index_best_fit=i
            self.best_individual=deepcopy(self.pop[self.index_best_fit])
            self.best_coeffs=deepcopy(coeffs[self.index_best_fit])
        return

    def best_rms(self,dataset):
        '''
        Returns root-mean-square error of best individual on given dataset
        '''
        if dataset=='train':
            x=self.x
            y=self.y
        elif dataset=='valid':
            x=self.x_v
            y=self.y_v
        elif dataset=='test':
            x=self.x_t
            y=self.y_t
        else:
            raise ValueError('Unknown dataset. Please chose form {train,valid,test}')

        poly=self.genome2poly(self.best_individual,x)

        y_predict=self.y_model(poly)(x,self.best_coeffs)
        rms=np.sqrt(np.mean((y-y_predict)**2))

        return rms

    def read_off_roulette_selections(self,cum_fit,ticks):
        '''
        Returns mates for breeding according to wheel tick positions after spinning roulette wheel
        '''
        mates=[]
        for j in range(0,self.pop_size):
            for i in range(0,self.pop_size):
                if cum_fit[i]>ticks[j]:
                    mates.append(i)
                    break
        return np.array(mates)

    def select_for_breeding(self):
        '''
        Returns individuals chosen for breeding using previously specified breeding procedure
        '''

        selected_mates=[]

        if self.select=='tournament':
            for i in range(self.pop_size):
                competitors=np.random.randint(0,self.pop_size,self.tourney_size)
                fitness=[self.pop_fit[j] for j in competitors]
                winner=competitors[np.argmax(fitness)]
                selected_mates.append(winner)

        if self.select=='roulette':
            prop_fit=self.pop_fit/sum(self.pop_fit)
            cum_fit=np.cumsum(prop_fit)
            x=np.random.rand() #spin wheel
            ticks=x+np.linspace(0,1,self.pop_size+1) #P equally spaced ticks
            ticks=ticks[:-1]-np.floor(ticks[:-1]) #stay in range (0,1)
            #align with fitness segments and read off selected mates
            selected_mates=self.read_off_roulette_selections(cum_fit,ticks)

        self.mates=selected_mates
        return

    def breed(self):
        '''
        Produces intermediate generation (after breeding, before mutation)
        '''
        pop_new=[]
        for i in range(self.pop_size/2):
            p1=self.pop[self.mates[i]]
            p2=self.pop[self.mates[i+1]]
            c1=[]
            c2=[]
            if np.random.rand() < self.recomb_prob:
                for p in [p1,p2]:
                    for term in range(self.terms):
                        #if one offspring is full append to other
                        if len(c1)==self.terms:
                            if p[term] not in c2:
                                c2.append(p[term])
                            else:
                                c2.append(self.rand_term())
                            continue
                        if len(c2)==self.terms:
                            if p[term] not in c1:
                                c1.append(p[term])
                            else:
                                c1.append(self.rand_term())

                            continue
                        #if space in both randomly assign as long as no duplicates occur
                        if np.random.rand()>0.5:
                            if p[term] not in c1:
                                c1.append(p[term])
                            else:
                                c2.append(p[term])

                        else:
                            if p[term] not in c2:
                                c2.append(p[term])
                            else:
                                c1.append(p[term])
                c1.sort()
                c2.sort()
                pop_new.append(c1)
                pop_new.append(c2)
            else:
                p1.sort()
                p2.sort()
                pop_new.append(p1)
                pop_new.append(p2)
        return pop_new

    def mutate(self):
        '''
        Mutates intermediate generation to prodcued offspring for next generation
        '''
        for i in range(self.pop_size):
           for j in range(1,self.terms):
                if np.random.rand()< self.mutate_prob:
                    while True:
                        new_term=self.rand_term()
                        if new_term not in self.pop[i]:
                            self.pop[i][j]=new_term
                            break
           self.pop[i].sort()
        self.pop.sort()
        return

    def save_poly(self,filepath='poly_out'):
        '''
        Saves the final best polynomial with optimised coefficients to file
        '''
        poly_out=[]
        for j in range(self.terms):
            temp=[self.best_coeffs[j]]
            for k in range(self.vars):
                temp.append(self.best_individual[j][k])
            poly_out.append(temp)

        np.savetxt(filepath,np.array(poly_out))

    def save_y_prediction(self,dataset,filepath='prediction_out'):
        '''
        Saves true y and predicted y for given dataset
        '''
        if dataset=='train':
            x=self.x
            y=self.y
        elif dataset=='valid':
            x=self.x_v
            y=self.y_v
        elif dataset=='test':
            x=self.x_t
            y=self.y_t
        else:
            raise ValueError('Unknown dataset. Please chose form {train,valid,test}')

        poly=self.genome2poly(self.best_individual,x)
        y_predict=np.squeeze(self.y_model(poly)(x,self.best_coeffs))
        y_out=[]
        for j in range(len(y)):
            y_out.append([y[j],y_predict[j]])

        np.savetxt(filepath,np.array(y_out))

    def run_GA(self,max_gens=500):
        '''
        Runs the GA to evolve population of polynomials

        Args:
            max_gens= maximum number of generations to evolve for
        '''
        try:
            self.x
        except:
            raise ValueError('Please load training data before running GA.')
        self.sol_index=hash_table()         #initialise hash_table to store optimised polynomials

        self.best_fit=0
        self.generate_pop()
        for i in range(max_gens):
            self.get_fitness()
            self.select_for_breeding()
            self.breed()
            self.mutate()
            if self.verbose:
                if i%10==0:
                    print 'Generation:', i,', Train RMS:',self.best_rms('train'),', Valid RMS',self.best_rms('valid')
