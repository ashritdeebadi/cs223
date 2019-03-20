
# coding: utf-8

# In[4]:


#Meenakshi Anbukkarasu
#Gayathri 

import random
from random import randint

def main():

filepath = "dna_sequences.dat"  
with open(filepath) as fp:  
    file_content = fp.readlines()
    mutation_type = ['M', 'N', 'I', 'D', 'U','R']
    for eachline in file_content:
        types=random.choice(mutation_type)
        if types=='U' or types=='R':
            integer=randint(0, 20)
            mutate(eachline,types,interger)
            crossover(eachline)
        
        else:
            mutate(eachline,types)
            crossover(eachline)

    
    
    
        

        
       

