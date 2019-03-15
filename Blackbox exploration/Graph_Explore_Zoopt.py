import numpy as np
import math
from zoopt import Dimension, Objective, Parameter, Opt, ExpOpt,Solution
#import networkx as nx
import pandas as pd
import time
import scipy.sparse as sci

for m in [5,6,7]: # number of battlefields
    n = int(m*3)
    #
    N = 2 + (n-1)*(m+1)       # Number of nodes
    E= int(2*(m+1) + (n-2)*(m+1)*(m+2)/2)   #number of edges
    P = int(math.factorial(m+n-1)/(math.factorial(m)*math.factorial(n-1)))    #number of paths
    
    ############ Useful graph functions #############
    def Layer(u):
        if u==0: 
            return int(0)
        elif u == N-1:
            return int(n)
        else:
            return int( np.floor((u-1)/(m+1))+1 )
    
    def Vertical(u):
        if (u==0): Vertical = 0
        elif (u ==N-1): Vertical = int(m)
        elif (np.remainder(u,m+1)==0): Vertical = m
        else: Vertical = np.remainder(u,m+1) -1
        return Vertical
    
    def Children(u):
        if u==0: children = np.array(range(1,m+2))
        elif (u >= N-1-(m+1)): children =np.array([N-1])
        else:   
            temp = range(0, m+1-Vertical(u))
            children = (u+m+1)*np.ones(m+1-Vertical(u))
            children = children + temp
        return children.astype(int)
    
    def node_to_edge(u1,u2):
        if (u1==0): edge = u2-1
        elif (u2==N-1): edge = E- (N-1-u1)
        else: edge =int( m +(Layer(u1)-1)*(m+2)*(m+1)/2 + (2*(m+1) - Vertical(u1)+1)*Vertical(u1)/2 +1 +(Vertical(u2)-Vertical(u1)))
        return edge
    
    def Ancestors(u):
        if u==N-1: ancestor = range(N-1)
        elif (Layer(u)<=1): ancestor = [0]
        else:
            ancestor = [0]
            for i in range(1,u-m): 
                if Vertical(i) <= Vertical(u): ancestor.append(i)
        return ancestor
    ############## Store a priory graph info ###################
    Layer_s = np.zeros(N)
    Vertical_s = np.zeros(N)
    Children_s = []
    Ancestor_s = []
    for u in range(N):
        Layer_s[u] = Layer(u)
        Vertical_s[u] = Vertical(u)
        Children_s.append(list(Children(u)))
        Ancestor_s.append(list(Ancestors(u)))
    Layer_s = Layer_s.astype(int)
    Vertical_s = Vertical_s.astype(int)
    #print(Layer_s, Vertical_s, Children_s,Ancestor_s)
    
    
    node_edge = sci.lil_matrix((N,N))
    edge_node = np.zeros((E,2))
    for u1 in range(N):
        for u2 in Children(u1):
            if (u1< N-1): 
                node_edge[u1,u2]= int(node_to_edge(u1,u2))
                edge_node[int(node_to_edge(u1,u2))] = [int(u1),int(u2)]
    
    
    #########################################################################
    
    start = time.time()
    ###########################################################################
    
    
    #### 
    def update_H(w):
        H = np.identity(N)
        for j in np.flip(range(N),axis=0):
            for i in np.flip(Ancestor_s[j], axis=0):
                for k in Children_s[i]:
                    H[i,j] = H[i,j]+ w[int(node_edge[i,k])]*H[k,j] 
        return H
    
    def single_prob(e,w,H): #probability that an edge e is chosen by exploiting distribution
        single_prob = H[0, int(edge_node[e,0])] * w[e] * H[int(edge_node[e,1]),N-1]/H[0,N-1]
        return single_prob
    #
    def coocurence_mat(w,H):
        mat = np.zeros((E,E))
        for e_1 in range(E):
            mat[e_1,e_1] = single_prob(e_1,w,H)
            for e_2 in range(e_1+1,E):
                mat[e_1,e_2] = H[0, int(edge_node[e_1,0])] * w[e_1] * H[int(edge_node[e_1,1]),int(edge_node[e_2,0])] *w[e_2] * H[int(edge_node[e_2,1]),N-1] / H[0,N-1]
                mat[e_2,e_1] = mat[e_1,e_2]                
        return mat
    #
    #
    #
    def smallest_eigen(solution):
        w = solution.get_x()
        for i in range(len(w)):
            if w[i] < low_bound : w[i] = low_bound
            if w[i] > up_bound: w[i] = up_bound
        H =update_H(w)   
        C_explore = coocurence_mat(w,H)
        eigenval = np.sort(np.round(np.linalg.eigvalsh(C_explore),8))
        for i in range(E):
            if np.real(eigenval[i])+0>0:
                lambda_min = np.real(eigenval[i])
                break
        return (-1)*lambda_min          # Maximize lambda_min = Minimize (-1*lambda_min)
    
    dim_size = E
    low_bound = 1e-5
    up_bound = 1
    
    dim = Dimension(dim_size, [[low_bound, up_bound]] * dim_size, [True] * dim_size )   #continuous serach from (0,1])
    objective = Objective(smallest_eigen, dim)
    
    #low_dim = 25
    #uniform = list([up_bound/2]*(low_dim +1))
    
    uniform = list([up_bound/2]*E)
    
    
    initial = [uniform]
    
    budget = E*100
    par = Parameter(budget=budget, intermediate_result=True,intermediate_freq=1000, init_samples = initial, precision = 1e-5)
    #par = Parameter(budget=budget, init_samples = initial, precision = low_bound, intermediate_result=True, intermediate_freq=1000,high_dim_handling=True, reducedim=True, num_sre=5, low_dimension=Dimension(low_dim, [[low_bound, up_bound]] * low_dim, [True] * low_dim))
    print('start')
    
    sol = Opt.min(objective, par)
    
    #print(sol.get_x(), sol.get_value())
    #solution_list = ExpOpt.min(objective, par, repeat=1, plot=True)
    #solution_list = ExpOpt.min(objective, par, repeat=5, best_n=5, plot=True, plot_file='Figure.pdf', seed=110)
    optimal = sol.get_x()
    optval= - sol.get_value()
    end = time.time()
    
    
    ######## PROJECT solution for LOW_DIM TECHNIQUES
    #for i in range(len(optimal)):
    #        if optimal[i] < low_bound : optimal[i] = low_bound
    #        if optimal[i] > up_bound: optimal[i] = up_bound
    #        
    #H = update_H(optimal)
    #C= coocurence_mat(optimal, H)
    #eigenval = np.sort(np.round(np.linalg.eigvalsh(C),8))
    #for i in range(E):
    #    if np.real(eigenval[i])+0>0:
    #        lambda_min = np.real(eigenval[i])
    #        break        
    #        
    
    df = pd.read_csv('explore.csv', header =None)		#choose output explore.csv
    df1 =pd.DataFrame(np.append([n,m,N,E,P,end-start,budget,optval],optimal))
    #df1 =pd.DataFrame(np.append([n,m,N,E,P,end-start,budget,lambda_min],optimal))   # For LOW_DIM TECHNIQUES
    output= pd.concat([df,df1], ignore_index=False, axis=1)
    output.to_csv('explore.csv',header =None, index= False)
    
    #### Test the solution
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
