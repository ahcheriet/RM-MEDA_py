#!/usr/bin/env python
#from openturns import FrankCopulaFactory,ClaytonCopulaFactory, NumericalSample, FrankCopula, ClaytonCopula
from PyGMO import algorithm,population,problem
import random as rd
import numpy as np
from math import sqrt
#import matplotlib.pylab as plt

from Generate import *


class rm_meda(algorithm.base):
   """
   A custom steady-state algorithm, based on the hypervolume computation.
   """

   def __init__(self, gen = 20, K = 5):
      """
      Constructs an instance of the algorithm.

      USAGE: rm_meda(gen=10, p_m=20)

      NOTE: Evolves the population using the least contributor feature.

      * gen: number of generations
      * n: the number of solutions to be sampled by the copula
      """
      # We start by calling the base constructor
      super(rm_meda,self).__init__()
      # Store the number of generations
      self.__gen = gen
      self.__K = K


   def get_all_vectors_and_fitness(self, pop):
        prob = pop.problem
        dim, cont_dim = prob.dimension, prob.dimension - prob.i_dimension
        lb, ub = prob.lb, prob.ub
        all_elements = []
        all_fitness = [] 
        lnp =len(pop)
        print lnp
        if len(pop.compute_pareto_fronts()[0]) != lnp:
            best_idx = pop.get_best_idx(lnp-len(pop.compute_pareto_fronts()[-1]))
        else:
            best_idx = pop.compute_pareto_fronts()[0]
        print best_idx
        for i in best_idx:
           all_elements.append(pop[i].cur_x)
           all_fitness.append(pop[i].cur_f)
        return list(zip(*all_elements)), all_elements, all_fitness

   def Modeling(self, pop):
        prob = pop.problem
        dim, cont_dim, n_obj = prob.dimension, prob.dimension - prob.i_dimension, prob.f_dimension
        lb, ub = prob.lb, prob.ub
        varr, List, List_fitness = self.get_all_vectors_and_fitness(pop)  # We get the variables and the objective values
        self.elements_array = np.array(List)            # Let us store the solutions and fitness values in arrays
        self.fitness_array = np.array(List_fitness) 
        return RMMEDA_operator(self.elements_array,self.__K,n_obj)

   def evolve(self, pop):
      if len(pop) == 0:
           return pop
      lnp =len(pop)
      prob = pop.problem
      dim, cont_dim, n_obj = prob.dimension, prob.dimension - prob.i_dimension, prob.f_dimension
      lb, ub = prob.lb, prob.ub
      # Main loop of the algorithm
      for s in range(self.__gen):
         print 'Gen',s
         self.genidx = s
         new_pop = self.Modeling(pop)
         for new_x in new_pop:
             try:
                pop.push_back(new_x)
             except ValueError: # we don't add the solution if it violates the constraints
                pass
         pf = pop.compute_pareto_fronts()
         while(len(pop) > lnp):
            pop.erase(pop.get_worst_idx())
         generation = 1
#         pop.plot_pareto_fronts()
#         plt.savefig('./IMG/fig'+str(s)+'.png')
         
  #       moea_instance_tmp =  algorithm.nsga_II(gen = 1)
  #       pop= moea_instance_tmp.evolve(pop)
      return pop


   def get_name(self):
         return "RM-MEDA Algorithm"

