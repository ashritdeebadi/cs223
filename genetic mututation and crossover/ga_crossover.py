
# coding: utf-8

# In[ ]:

import random
from random import randint
import string


def main():
    """
    Starts the main sequence of operations
    """
    #count = 0

    filepath = "dna_sequences.dat"  

    with open(filepath,'r') as fp:  

        file_content = fp.readlines()
    
    fp2 = open('crossover_results.dat', 'w+')
 
    for eachline in file_content:

        #count = count + 1 

        #print (count)

        crossover_result = crossover(str(eachline).rstrip())

        crossover_result = '({},{},{}),'.format(crossover_result[0], crossover_result[1], crossover_result[2])

        fp2.write(str(crossover_result))
            
    fp.close()

    fp2.close()


def crossover(dna):

    length_dna = len(dna) 

    if (length_dna <= 4 ):
        
        location = randint(0, length_dna-1)

        length = randint(1,1)

    if(length_dna == 1 ):

        location = 0

        length = 1

    if (length_dna > 4 and length_dna <= 9):

        location = randint(0, length_dna-4)

        length = randint(1,2)

    if (length_dna > 9  ):

        location = randint(0, length_dna-10)

        length = randint(1,7)


    trans = str.maketrans('ATGC', 'TACG')

    complement=dna.translate(trans)

    end=location+length

    replace=complement[location:end]

    crossed_dna=dna[:location-1]+replace+dna[end:]

    return (crossed_dna,location,length)

if __name__== "__main__":
  main()

