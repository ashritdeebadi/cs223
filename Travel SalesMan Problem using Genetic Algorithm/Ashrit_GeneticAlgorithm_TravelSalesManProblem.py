
# coding: utf-8

# In[9]:


import random

from deap import base
from deap import creator
from deap import tools
import pandas as pd
import numpy as np
 
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

IND_SIZE= 8
toolbox.register("pop_generator", random.sample,range(IND_SIZE),IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, 
    toolbox.pop_generator)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# please change this path to the file
distances = pd.read_csv("C:/Users/user/Desktop/223/dataset_inprog_1.csv")
print(distances)

def cal_distance(individual):
    dis=[]
    
    
    for i in range(len(individual)-1):    
        new_dis=distances.iloc[individual[i],individual[i+1]]
        dis.append(new_dis)
    return sum(dis),

def main():
    random.seed(100)
    pop = toolbox.population(n=50)
    print(pop)
    toolbox.register("evaluate", cal_distance)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    
    
    sel= 0.3
    mut=0.5
    print('start')
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    print("  Evaluated %i individuals" % len(pop))    
    
     
    fits = [ind.fitness.values[0] for ind in pop]
    
    g = 0
    

    with open("Ashrit_GA_TS_info.txt", "w") as text_file:
        while min(fits) > 16927 and g < 1000:
        
            g = g + 1
            print("-- Generation %i --" % g)
        
       
            offspring = toolbox.select(pop, len(pop))
        
            offspring = list(map(toolbox.clone, offspring))
        
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < sel:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mut:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
        
            fits = [ind.fitness.values[0] for ind in pop]
        
            length = len(pop)
            mean = sum(fits) / length
            median = np.median(fits)
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
        
        
            print("  Length of population %s" %length)
            print("  Avg %s" % mean)
            print("  Median %s"% median)
            print("  Std %s" % std)
            print("--End of (successful) evolution --")
    
            best_ind = tools.selBest(pop, 1)[0]
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            
            print("  Length of population %s" %length,file=text_file)
            print("  Avg %s" % mean,file=text_file)
            print("  Median %s"% median,file=text_file)
            print("  Std %s" % std,file=text_file)
            print("--End of (successful) evolution --",file=text_file)
            print("  Best individual is %s, %s" % (best_ind, best_ind.fitness.values),file=text_file)
            print("",file=text_file)
            for i in range(len(best_ind)):
                print()
        dictionary={0:'London',1:'Venice',2:'Dunedin',3:'Singapore',4:'Beijing',5:'Phoenix',6:'Tokyo',7:'Victoria'}
        path=["first","second","third","fourth","fifth","sixth","seventh","eighth"]
        with open("Ashrit_GA_TS_Result.txt","w") as text_file:
            n=0
            for i in best_ind:
                print( " %s Name of the %s city visited" %(dictionary[i] ,path[n]),file=text_file)
                n+=1
            

if __name__ == "__main__":
    main()

