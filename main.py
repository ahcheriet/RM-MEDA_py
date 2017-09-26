from RMMEDA import *
import matplotlib.pylab as plt



#prob = problem.zdt(1) # creating a problem here we used zdt1
#prob = problem.dtlz(3)
prob = problem.kur() 
#prob = problem.fon() 
#prob = problem.cec2009(1)
#prob = problem.cassini_1(objectives=2) 
pop = population(prob,100)


alg = rm_meda(gen = 20, K = 5)
pop = alg.evolve(pop)
pop.plot_pareto_fronts() # ploting pareto fronts
#prob.plot(pop1)
plt.show() 



