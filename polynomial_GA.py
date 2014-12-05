__author__ = 'Robert Hogan'

from hash_table import *
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy

class polyGA:
    def __init__(self,pop_size=100, mutate_prob=0.03,recomb_prop=0.5,poly_degree=5,terms=20,vars=5, selection='roulette',tournmanent_size=None):
        if selection=='tournament' and not tournmanent_size:
            raise ValueError('Please input desired tournament size when using tournament selection')
            return
        if selection not in['tournament','roulette']:
            raise ValueError('Please choose valid selection method')
            return

        self.select=selection               #set selection method
        self.tourney_size=tournmanent_size  #set tournament size
        self.pop_size=pop_size              #set size of population
        self.mutate_prob=mutate_prob        #set mutation probability
        self.recomb_prob=recomb_prop        #set recombination probability
        self.degree=poly_degree             #set maximum degree of polynomial
        self.terms=terms                    #set number of terms in polynomial
        self.vars=vars                      #set number of variables in polynomial
        self.sol_index=hash_table()         #initialise hash_table to store optimised polynomials
        return

    def load_training_data(self,xdata,ydata):
        self.x=xdata
        self.y=ydata
        return

    def load_validation_data(self,xdata,ydata):
        self.x_v=xdata
        self.y_v=ydata
        return

    def load_test_data(self,xdata,ydata):
        self.x_t=xdata
        self.y_t=ydata
        return

    def rand_term(self):
        term=[]
        total=0
        max_power=np.random.randint(0,self.degree+1)
        for i in range(0,self.vars):
            if total==max_power:
                term.append(0)
                continue
            n=np.random.randint(0,max_power-total+1)
            total=total+n
            term.append(n)
        np.random.shuffle(term)

        return term

    def generate_pop(self):
        pop=[]
        for i in range(0,self.pop_size):
            individual=[]
            individual.append([0]*self.vars)
            for j in range(1,self.terms):
                while True:
                    term=self.rand_term()
                    if term not in individual:
                        break
                individual.append(term)
            individual.sort()
            pop.append(individual)
        self.pop=pop
        return

    def genome2poly(self,individual,x):
        #for each term of individual raise data to powers specified
        t=np.array([x**np.array(individual[i]) for i in range(len(individual))])
        #muliply together subterms e.g. [x1^2,x2^0]->x1^2*x2^0
        p=t.prod(axis=2)
        return p

    def y_cf(self,poly):
        x=self.x
        def y_cf_i(x,*coeffs):
            #muliply by coeffs and sum to get predicted y value for each data point
            y=np.dot(coeffs,poly)
            return y
        return y_cf_i

    def jacobian_cf(self,poly):
        def jacobian_cf_i(*coeffs):
            return np.array(poly)
        return jacobian_cf_i

    def optimize_individual(self,individual):
        init_coeffs=2*np.random.rand(len(individual))-1.0
        poly=self.genome2poly(individual,self.x)
        coeffs=curve_fit(self.y_cf(poly),self.x,self.y,p0=init_coeffs,Dfun=self.jacobian_cf(poly),col_deriv=1)[0]
        residuals=np.squeeze(self.y_cf(poly)(self.x,coeffs))-self.y
        err=np.dot(residuals,residuals)
        fit=1.0/err

        return coeffs,fit

    def get_fitness(self):
        fit=[]
        coeffs=[]
        for individual in self.pop:
            out=False
            if self.sol_index.size !=0:
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
        i=np.argmax(fit)
        if fit[i]>self.best_fit:
            self.best_fit=fit[i]
            self.index_best_fit=i
            self.best_individual=deepcopy(self.pop[self.index_best_fit])
            self.best_coeffs=deepcopy(coeffs[self.index_best_fit])
        return
    def best_rms(self):
        poly=self.genome2poly(self.best_individual,self.x)

        y_predict=self.y_cf(poly)(self.x,self.best_coeffs)
        self.rms=np.sqrt(np.mean((self.y-y_predict)**2))
        return self.rms

    def read_off_roulette_selections(self,cum_fit,ticks):
        mates=[]
        for j in range(0,self.pop_size):
            for i in range(0,self.pop_size):
                if cum_fit[i]>ticks[j]:
                    mates.append(i)
                    break
        return np.array(mates)

    def select_for_breeding(self):
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
        for i in range(self.pop_size):
           for j in range(1,self.terms):
                if np.random.rand()< self.recomb_prob:
                    while True:
                        new_term=self.rand_term()
                        if new_term not in self.pop[i]:
                            self.pop[i][j]=new_term
                            break
           self.pop[i].sort()
        self.pop.sort()
        return

    def save_poly(self,filepath='poly_out'):
        poly_out=[]
        for j in range(self.terms):
            temp=[self.best_coeffs[j]]
            for k in range(self.vars):
                temp.append(self.best_individual[j][k])
            poly_out.append(temp)

        np.savetxt(filepath,np.array(poly_out))

    def save_y_prediction(self,dataset,filepath='prediction_out'):
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
        y_predict=np.squeeze(self.y_cf(poly)(x,self.best_coeffs))
        y_out=[]
        for j in range(len(y)):
            y_out.append([y[j],y_predict[j]])

        np.savetxt(filepath,np.array(y_out))

    def run_GA(self,max_gens=500):
        try:
            self.x
        except:
            raise ValueError('Please load training data before running GA.')
        self.best_fit=0
        self.generate_pop()
        for i in range(max_gens):
            self.get_fitness()
            self.select_for_breeding()
            self.breed()
            self.mutate()
            print i,self.best_rms()
