import cvxpy as cp
import numpy as np
import sympy as sp
import scipy.sparse as sci
import networkx as nx
import pandas as pd
#import matplotlib.pyplot as plt
import time
###########################################################################
###########################################################################
# INPUT of the game
m=4
n=3
N = 2 + (n-1)*(m+1)       # Number of nodes
E= int(2*(m+1) + (n-2)*(m+1)*(m+2)/2)   #number of edges
P = int(sp.factorial(m+n-1)/(sp.factorial(m)*sp.factorial(n-1)))    #number of paths

##########################################################################
##########################################################################
# DEFINE the graph and find all (binary) paths
#Define the graph
G=nx.DiGraph()
G.add_nodes_from(range(0,N))        #add N nodes, indexed by 0 to N-1


#Define a function to compute the layer of a node
def Layer(u):
    if u==0: 
        return int(0)
    elif u == N-1:
        return int(n)
    else:
        return int( np.floor((u-1)/(m+1))+1 )

###############    
# Add edges from 0 to 1st-layer nodes
for j in range(1,m+2):      
    #Trans_matrix[0,j] =1    
    G.add_edge(0,j)
# Add edges from (n-1)th-layer nodes to N-1
for i in range(1+(n-2)*(m+1), N-1): 
    #Trans_matrix[i,N-1]=1     
    G.add_edge(i,N-1)
    
u = 1
while u < N-1-(m+1):
    for j in range(u+m+1,int((Layer(u)+1)*(m+1) + 1)):
        #Trans_matrix[u,j]=1
        G.add_edge(u,j)
    u += 1
    
##############    
# Draw the graph
#nx.draw(G, pos=nx.shell_layout(G), with_labels=True)

#The sets of nodes, edges, paths (series of nodes)
nodes = list(G.nodes())
edges = list(G.edges())
#print(edges)
paths_by_nodes = list(nx.all_simple_paths(G, source=0, target=N-1))
#print(paths_by_nodes) 
       
##############
#TRANSCRIPT paths into vectors
#A function to translate a path into a sequence of nodes:
def edgewise(p):
    path_temp = np.zeros(shape=(n,2))
    for j in range(0,n):
       path_temp[j]=[p[j],p[j+1]]     
    return path_temp

#Translate into binary vectors
bin_paths = sci.lil_matrix((P, E)) # matrix to store all bin_vectors of all paths
for i in range(0,P):
    for j in range(0,E):
        if any(np.equal(edgewise(paths_by_nodes[i]),edges[j]).all(1))==1:   #if the j-th edge is in the path[i] (edgewise)
            bin_paths[i,j]=1
#print(bin_paths)
            
###########################################################################
###########################################################################
            # OPTIMIZATION OF THE SMALLEST EIGEN_VALUE => EXPLORATION DISTRIBUTION
# CONSTRUCT the problem.

#define the function of covariance matrix
def f(p): 
    return (p.transpose()).dot(p)


# Find the mulplicity of zero-eigenvalue:
Uniform = sci.lil_matrix((E, E))
Uniform = (1/P)*f(bin_paths[0,])
for i in range(1,P):
    Uniform+= (1/P)*f(bin_paths[i,])
    
Rank = np.linalg.matrix_rank(Uniform.todense(), hermitian=True)
K = E - Rank
##################
print('Start to solve SDP')
start = time.time()

s= cp.Variable()
x= cp.Variable(P)
Z= cp.Variable((E, E), PSD=True)


M = x[0]*f(bin_paths[0,])
for i in range(1,P):
    M+= x[i]*f(bin_paths[i,])
    
objective = cp.Minimize( (K+1)*s + cp.trace(Z) )  #maximize the sum of (E-Mul_zero+1) smallest eigen values (mulplicity of eigenvalue 0 is E-Mul_zero)
#objective = cp.Maximize(cp.lambda_min(M))

constr_1 = [sum(x) ==1]
constr_2 = [x >= 10**(-15)] #x greater than zeros since distribution spanned by S
constr_4 = [(Z + M + s*np.eye(E)) >> 0]
#constr_5 = [Z >> 0]
constraints = constr_1 + constr_2 + constr_4 

prob = cp.Problem(objective, constraints)
end = time.time()
############
#Solve the SDP problem to maximize smallest positive eigen value of matrix M
# The optimal objective value is returned by `prob.solve()`.
eigen_lambda = - prob.solve(solver=cp.CVXOPT,verbose=True)
# The optimal value for x is stored in `x.value`.
distri_mu = np.array(x.value) 
print(distri_mu)
print(eigen_lambda)

    

#            
    
df = pd.read_csv('Test_SDP.csv', header =None)	#Output to Test_SDP.csv
df1 =pd.DataFrame(np.append([n,m,N,E,P,end-start,eigen_lambda],distri_mu))
#df1 =pd.DataFrame(np.append([n,m,N,E,P,end-start,budget,lambda_min],optimal))   # For LOW_DIM TECHNIQUES
output= pd.concat([df,df1], ignore_index=False, axis=1)
output.to_csv('Retest_SDP.csv',header =None, index= False)
    
    ### Test the solution
    #Location = 'result.csv' #Change this for the location of data file
    #data= pd.read_csv(Location, header=None, usecols=[3]) #columns resp. to the instance
    #data = np.array(data.dropna())
    #distri_mu = (data[8:(E+8)]).flatten()
    
    #H = update_H(optimal)
    #C= coocurence_mat(optimal, H)
    #eigenval = np.sort(np.round(np.linalg.eigvalsh(C),8))
    #for i in range(E):
    #    if np.real(eigenval[i])+0>0:
    #        lambda_min = np.real(eigenval[i])
    #        break
