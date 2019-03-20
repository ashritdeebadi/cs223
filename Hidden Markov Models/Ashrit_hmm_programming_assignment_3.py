
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
#matplotlib inline

"""
A Markov chain (model) describes a stochastic process where the assumed probability 
of future state(s) depends only on the current process state and not on any the states 
that preceded it (shocker).

Let's get into a simple example. Assume you want to model the future probability that 
your dog is in one of three states given its current state. To do this we need to 
specify the state space, the initial probabilities, and the transition probabilities.

Imagine you have a very lazy fat dog, so we define the state space as sleeping, eating, 
or pooping. We will set the initial probabilities to 35%, 35%, and 30% respectively.
"""

# create state space and initial state probabilities

states = ['a', 'c','g','t']
pi = [0.25, 0.25, 0.25,0.25]
state_space = pd.Series(pi, index=states, name='states')
print(state_space)
print(state_space.sum())
print()

# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

q_df = pd.DataFrame(columns=states, index=states)
#q_df.loc[states[0]] = [0.4, 0.2, 0.4]
#q_df.loc[states[1]] = [0.45, 0.45, 0.1]
#q_df.loc[states[2]] = [0.45, 0.25, .3]

q_df.loc[states[0]] = [0.180, 0.274, 0.426, 0.120]
q_df.loc[states[1]] = [0.170, 0.368, 0.274, 0.188]
q_df.loc[states[2]] = [0.161, 0.339, 0.375, 0.125]
q_df.loc[states[3]] = [0.079, 0.355, 0.384, 0.182]

print(q_df)

q = q_df.values
print('\n', q, q.shape, '\n')
print(q_df.sum(axis=1))
print()


"""
Now that we have the initial and transition probabilities setup we can create a 
Markov diagram using the Networkx package.

To do this requires a little bit of flexible thinking. Networkx creates Graphs 
that consist of nodes and edges. In our toy example the dog's possible states are 
the nodes and the edges are the lines that connect the nodes. The transition 
probabilities are the weights. They represent the probability of transitioning 
to a state given the current state.

Something to note is networkx deals primarily with dictionary objects. With that 
said, we need to create a dictionary object that holds our edges and their weights.
"""

from pprint import pprint 

# create a function that maps transition probability dataframe 
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(q_df)
print('''edges_wts is ''')
pprint(edges_wts)
print()

"""
Now we can create the graph. To visualize a Markov model we need to 
use nx.MultiDiGraph(). A multidigraph is simply a directed graph which can have 
multiple arcs such that a single node can be both the origin and destination. 

In the following code, we create the graph object, add our nodes, edges, and 
labels, then draw a bad networkx plot while outputting our graph to a dot file. 
"""

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
states = ['a', 'c']
G.add_nodes_from(states)
#print(f'Nodes:\n{G.nodes()}\n')
print('Nodes:\n', G.nodes(), "\n")

# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
#print(f'Edges:')
print("g.edge is Edges:")
pprint(G.edges(data=True))    

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)
# In Windows: dot -Tps filename.dot -o outfile.ps


# create edge labels for jupyter plot but is not necessary
edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'pet_dog_markov.dot')
print()

print("========================================================================")
print("                   NOW HMMs")
print("========================================================================") 
print()

"""
Consider a situation where your dog is acting strangely and you wanted to model 
the probability that your dog's behavior is due to sickness or simply quirky 
behavior when otherwise healthy.

In this situation the true state of the dog is unknown, thus hidden from you. 
One way to model this is to assume that the dog has observable behaviors that 
represent the true, hidden state. Let's walk through an example.

First we create our state space - healthy or sick. We assume they are equiprobable.  
"""

# create state space and initial state probabilities

#hidden_states = ['healthy', 'sick']
#pi = [0.5, 0.5]

hidden_states = ['I', 'N']
pi = [0.3, 0.7]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())
print()

# Next we create our transition matrix for the hidden states. 
# create hidden transition matrix
# a or alpha = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.6, 0.4]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

print(a_df)

a = a_df.values
print('\n', a, a.shape, '\n')
print(a_df.sum(axis=1))
print()

"""
This is where it gets a little more interesting. Now we create the emission or 
observation probability matrix. This matrix is size M x O where M is the number 
of hidden states and O is the number of possible observable states. 

The emission matrix tells us the probability the dog is in one of the hidden 
states, given the current, observable state. 

Let's keep the same observable states from the previous example. The dog can be 
either sleeping, eating, or pooping. For now we make our best guess to fill in 
the probabilities. 
"""

# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states 
# and O is number of different possible observations

#observable_states = ['sleeping', 'eating', 'pooping']
observable_states = ['a', 'c', 'g','t']
print()
print("observable_states:\n", states)
print("hidden_states:\n", hidden_states)
print()

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.4, 0.2,0.4]
b_df.loc[hidden_states[1]] = [0.3, 0.2, 0.3,0.2]


print(b_df)

b = b_df.values
print('\n', b, b.shape, '\n')
print(b_df.sum(axis=1))
print()


# Now we create the graph edges and the graph object. 
# create graph edges and weights

hide_edges_wts = _get_markov_edges(a_df)
pprint(hide_edges_wts)

emit_edges_wts = _get_markov_edges(b_df)
pprint(emit_edges_wts)
print()

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(hidden_states)
#print(f'Nodes:\n{G.nodes()}\n')
print('Nodes:\n', G.nodes(), '\n')

# edges represent hidden probabilities
for k, v in hide_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

# edges represent emission probabilities
for k, v in emit_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    
print('Edges:')
pprint(G.edges(data=True))    

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
emit_edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=emit_edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'pet_dog_hidden_markov.dot')
# In Windows: dot -Tps filename.dot -o outfile.ps

print()

"""
The hidden Markov graph is a little more complex but the principles are the same. 
For example, you would expect that if your dog is eating there is a high 
probability that it is healthy (60%) and a very low probability that the dog is 
sick (10%).

Now, what if you needed to discern the health of your dog over time given a 
sequence of observations?  
"""

# observation sequence of dog's behaviors
# observations are encoded numerically

#obs_map = {'sleeping':0, 'eating':1, 'pooping':2}
#obs = np.array([1,1,2,1,0,1,2,1,0,2,2,0,1,0,1])

#obs_map = {'A':0, 'C':1, 'G':2,'T':3}
#obs_seq=[]
#intseq='cggtgaaactgcacgattg'
#for i in range(len(intseq)):
#    obs_seq.append(i)
    
#obs = np.array([1,1,2,1,3,0,1,3,2,1,3,0,2,2,3,0,1,0,1])

#inv_obs_map = dict((v,k) for k, v in obs_map.items())
#obs_seq = [inv_obs_map[v] for v in list(obs)]

#obs_seq=['C', 'C', 'G', 'C', 'T', 'A', 'C', 'T', 'G', 'C', 'T', 'A', 'G', 'G', 'T', 'A', 'C', 'A', 'C']
#print('''''')
#print(obs_seq)
#print('''''''')
print('------------------')
my_obs_seq=[]
input_sequence='cggtgaaactgcacgattgttgctggcttaaagatagaccaatcagagtgtgtaacgtcatatttagcgtcttctatcatccaatcactgcactttacacactataaatagagcagctcatgggcgtatttgcgctagtgttgggtgttccgctgtgctgtttttccgtcatggctcgcactaagcaaactgctcggaagtctactggtggcaaggcgccacgcaaacagttggccactaaggcagcccgcaaaagcgctccggccaccggcggcgtgaaaaagccccaccgctaccggccgggcaccgtggctctgcgcgagatccgccgttatcagaagtccactgaactgcttattcgtaaactacctttccagcgcctggtgcgcgagattgcgcaggactttaaaacagacctgcgtttccagagctccgctgtgatggctctgcaggaggcgtgcgaggcctacttggtagggctatttgaggacactaacctgtgcgccatccacgccaagcgcgtcactatcatgcccaaggacatccagctcgcccgccgcatccgcggagagagggcgtgattactgtggtctctctgac'
for i in range(len(input_sequence)):
    my_obs_seq.append(input_sequence[i])
print(my_obs_seq)
my_dict = {0:'a', 1:'c', 2:'g',3:'t'}
my_obvs=np.array(my_obs_seq)
my_obvs

inv_obs_map_1 = dict((v,k) for k, v in my_dict.items())
obs_seq_1 = [inv_obs_map_1[v] for v in list(my_obvs)]
obs_seq_1
obs=np.array(obs_seq_1)
obs
obs_map = {'a':0, 'c':1, 'g':2,'t':3}
inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]
obs_seq

print( pd.DataFrame(np.column_stack([obs, obs_seq]), 
                columns=['Obs_code', 'Obs_seq']) )
print()


"""
Using the Viterbi algorithm we can identify the most likely sequence of hidden 
states given the sequence of observations.

High level, the Viterbi algorithm increments over each time step, finding the 
maximum probability of any path that gets to state iat time t, that also has 
the correct observations for the sequence up to time t.

The algorithm also keeps track of the state with the highest probability at 
each stage. At the end of the sequence, the algorithm will iterate backwards 
selecting the state that "won" each time step, and thus creating the most likely 
path, or likely sequence of hidden states that led to the sequence of 
observations.
"""

# define Viterbi algorithm for shortest path code adapted from Stephen 
# Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1])) #LPW
    for t in range(T-2, -1, -1): 
        path[t] = phi[int(path[t+1]), [t+1]]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1])) #LPW
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

path, delta, phi = viterbi(pi, a, b, obs)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)
print()

# Let's take a look at the result. 
state_map = {0:'I', 1:'N'}
state_path = [state_map[v] for v in path]

print()
print("RESULT:")

a=np.transpose(state_path)
str1 = ''.join(state_path)
results=[[input_sequence],[str1]]
print(results)

 


"""
References

    https://en.wikipedia.org/wiki/Andrey_Markov
    https://www.britannica.com/biography/Andrey-Andreyevich-Markov
    https://www.reddit.com/r/explainlikeimfive/comments/vbxfk/eli5_brownian_motion_and_what_it_has_to_do_with/
    http://www.math.uah.edu/stat/markov/Introduction.html
    http://setosa.io/ev/markov-chains/
    http://www.cs.jhu.edu/~langmea/resources/lecture_notes/hidden_markov_models.pdf
    https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
    http://hmmlearn.readthedocs.io
    http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
"""

