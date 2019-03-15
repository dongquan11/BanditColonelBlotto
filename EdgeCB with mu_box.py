import numpy as np
import math
import scipy.sparse as sci
#import networkx as nx
#import matplotlib.pyplot as plt
import time
import pandas as pd
import random 
# INPUT of the game

n=6  # number of battlefields
m=3    # number of troops of LEARNER
p=m  # number of troops of ADVERSARY
Time_range = [20000,30000]   #Choose the Time Horizon T
save = 'output.csv'      #Choose the output file

# Generate uniformly the battlfields values
#temp_val= np.random.uniform(low=1, high=8, size=n)  
#values= temp_val/sum(temp_val)
values = np.array([0.19111493, 0.11137238, 0.1715812 , 0.25752549, 0.19212109,0.07628491])	#Test with fixed battlefields values
# Battlefields values for the extreme_strong adversary
#    values = 0.01*np.ones(n)
#    values[n-1] = 1-(np.sum(values)-0.01)


setting = 'Bandit'  										# Feedback setting
exploration ='Uniform'     									# Either 'Uniform' exploration Or 'Optimal' exploration
Col_opt = int(11)    										# Change this to column of result.csv that corresponds to each instance
adv_distri = ['Battlefield_wise', 'Uniform_1']            	# Choose adversary's strategy: ether 'Battlefield_wise', 'Uniform_1' or 'Test_extreme'
gamma_option = "Bianchi"             						# Parameter tuning according to [Cesa Bianchi' 2012] 
       
    # BLOTTO rule
def blotto(x,y):
    if (x > y):  return 1
    elif (x < y): return 0
    else: return (1/2)
#
N = 2 + (n-1)*(m+1)  									    # Number of nodes
E= int(2*(m+1) + (n-2)*(m+1)*(m+2)/2)   					# Number of edges
P = int(math.factorial(m+n-1)/(math.factorial(m)*math.factorial(n-1)))    # Number of paths


########################    GRAPH INFO FUNCTIONS    #######################
###########################################################################
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
############## Storing the graph info ###################
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
			
#Define some other useful functions			
            
def allo(e):       											# The allocation corresponding to an edge e
    alloc = np.zeros(2)	
    alloc[1] =  Layer(edge_node[e,1])					    # The battlefield corresponds to edge e
    if (edge_node[e,0]==0) or (edge_node[e,1]==N-1) :    	# if the edge comes from source 0 or to root N-1 
        alloc[0]= edge_node[e,1] - edge_node[e,0] -1  		# allocate alloc[0] to battlefield alloc[1]
    else:
        alloc[0] = edge_node[e,1]- edge_node[e,0] - (m+1)
    return alloc
                

def bin_path(p):           #Translate a paths_by_nodes into a binary path \in {0,1}^E
    path_temp = np.zeros(shape=(n,2))
    for i in range(0,n):
       path_temp[i]=[p[i],p[i+1]]     
    
#    print(path_temp)
    bin_paths = np.zeros(E)
    for j in range(0,E):
        if any(np.equal(path_temp,edge_node[j]).all(1))==1:     #if the j-th edge is in the path_temp (edgewise)
            bin_paths[j]=1
    return bin_paths
##############################################################################
##############################################################################
##########################   WEIGHT PUSHING TECHNIQUE    ######################


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
########################################################################
##################   STRATEGY of ADVERSARY       ########################
def adversary(option):
    adver_stra = np.zeros(n)
    if (option == 'Uniform_1'):
        battle = np.array(range(1,n+1))
        resource = p
        while (resource > 0 and battle.size>1):
            b =  np.random.choice(battle, size=1, replace=False)     #draw uniformly a battlefield
            allo = random.randint(0,resource) #Draw uniformly an allocation
            resource = resource - allo
            adver_stra[b-1]= allo
            battle = np.delete(battle,np.where(battle==np.array(b)))
            #print([b, allo],battle,resource,'\n')
            
        adver_stra[battle-1]=resource
    elif (option == 'Battlefield_wise'):
        resource = p
        while (resource > 0):
            b =  np.random.choice(np.array(range(1,n+1)), size=1, p = values/sum(values))     #draw a battlefield based on values
            adver_stra[b-1] += 1
            resource -= 1
        
    elif ( option == 'Test_extreme'):
        adver_stra = (m+1)*np.ones(n) # Adversary puts m+1 too all battlefields (he guarantee to win)
        adver_stra[-1] = m-1    #The ast battlefield is allocated m-1 by adversary
        # The best (and only) option for learning is putting m troops to last battlefield
        
    return adver_stra
#print(adversary('uniform_1'))
#print(adversary('battlefield_wise'))
#adver_stra= adversary('battlefield_wise')

##############################################################################
#########        LOSS generated by the Adversary
def Loss(adver_stra):       #A E-dim vector that is the loss of each egde comparing with adver_stra
    L = np.zeros(E)
    for e in range(0,E):
        L[e] = values[int(allo(e)[1])-1] - values[int(allo(e)[1])-1]*blotto(allo(e)[0], adver_stra[int(allo(e)[1]) - 1]) #allo[0]=allocation to allo[1] = battlefield
    return L   
#print(Loss(adver_stra))  

#####################################################################################################
#####################################################################################################
######################################################################################################
###########                           EDGECB_ALGORITHM                             #########################



################# EXPLOIT using w_edges and H = Sample a path by w_edges and H #################
def exploit(H,w):
    node_k_1 = 0
    chosen_path = np.array([0])
    while (len(chosen_path) <= n):
        prob = np.array([])
        for k in Children_s[node_k_1]:
            prob = np.append(prob, [w[int(node_edge[node_k_1,k])] * H[k,N-1] / H[node_k_1,N-1] ])
#        print(prob)
        node_k = np.asscalar(np.random.choice(Children_s[node_k_1], size =1, p =prob  ))
#        print(node_k)
        chosen_path = np.append(chosen_path, node_k)
        node_k_1 = node_k
    
    return chosen_path
    
    
#######################################################################
##################    EXPLORATION       ###############################

if exploration == 'Uniform':
    w_uniform = np.ones(E)
    H_uniform = update_H(w_uniform)
    C_explore = coocurence_mat(w_uniform,H_uniform)
    eigenval = np.sort(np.round(np.linalg.eigvalsh(C_explore),8))
    for i in range(E):
        if np.real(eigenval[i])+0>0:
            lambda_min = np.real(eigenval[i])
            break
    Prob_explore = []
    for u in range(N-1):
        prob =[]
        for k in Children_s[u]:
            prob = np.append(prob, [w_uniform[int(node_edge[u,k])] * H_uniform[k,N-1] / H_uniform[u,N-1] ])
        Prob_explore.append(list(prob))
           
elif exploration == 'Optimal':    
    #Input of the optimal exploration distribution
    Location = 'explore.csv' #Change this for the location of data file
    data= pd.read_csv(Location, header=None, usecols=[Col_opt]) #columns resp. to the instance
    data = np.array(data.dropna())
    lambda_min = np.asscalar(data[7])
    w_opt = (data[8:(E+8)]).flatten()
    H_opt = update_H(w_opt)
    C_explore = coocurence_mat(w_opt,H_opt)
    Prob_explore = []
    for u in range(N-1):
        prob =[]
        for k in Children_s[u]:
            prob = np.append(prob, [w_opt[int(node_edge[u,k])] * H_opt[k,N-1] / H_opt[u,N-1] ])
        Prob_explore.append(list(prob))
#################################        
def explore():    
    node_k_1 = 0
    chosen_path = np.array([0])
    while (len(chosen_path) <= n):
        node_k = np.asscalar(np.random.choice(Children_s[node_k_1], size =1, p =Prob_explore[node_k_1] ))
#        print(node_k)
        chosen_path = np.append(chosen_path, node_k)
        node_k_1 = node_k
    return chosen_path  

#####################################################################
        ################## LOST ESTIMATION ##############
######################################################################
def est_loss(path, w,H,option,loss):
    estimate = np.zeros(E)      #Estimate loss for each edge
    if option == 'Bandit':
#        bandit_loss = np.dot(Loss(adver_stra),bin_path(path))
        C = (1-gamma)*coocurence_mat(w,H)  + gamma *C_explore
#        C = coocurence_mat(w,H)
        estimate = np.asarray(loss *  (np.matmul(np.linalg.pinv(C), bin_path(path)))).flatten()
    else:
        print('error in the name of bandit setting')
    return estimate
    
    ######################################################################
        ##############################################################
             #########  STARTING FOR LOOP   #####################
for fois in [1]:                
    for adv_distribution in adv_distri:    # Adversary plays 'Battlefield_wise' or 'Uniform_1' or 'Test_extreme'        
        for T in Time_range :
                             
            # Parameters of EXP.CB algorithm:
            B = np.sqrt(n)
            if gamma_option == "Bianchi":
                gamma = (B/lambda_min) * np.sqrt( np.log(P) / (T*( (E/(B**2)) + (2/lambda_min) )) ) 
            elif gamma_option =="One":
                gamma = 1
            else: print("Error in gamma option")
            
            eta = gamma*lambda_min/(B**2)        #Each action is an (0,1) vectors with dim=E and exactly n numbers 1
    
            start = time.time()
          
            #######################################################################################################
            #############################          MAIN ALGORITHM          ########################################
            #######################################################################################################
            # Initialization
            adv_store = np.zeros((T+1,n)) #A matrix to store all adversary strategies
            Cul_loss= 0
            w_edges = np.ones(E)
            H = np.identity(N)
            chosen_path = np.zeros(E)
            
            
            for t in range(1,T+1):
                adver_stra= adversary(adv_distribution)  # Adversary plays an action: Option= 'Battlefield_wise' or 'Uniform_1'
                adv_store[t] = adver_stra
    #            gamma = (t**(-1/2))/2   #Sasuke decreasing parameters
    #                print('gamma = ', gamma)
                Coin = np.random.choice([0,1], size=1, p = [1-gamma, gamma])     # Flip a coin, bias gamma
            #    Coin = 0
                if (setting == 'Full_info'): Coin =0
                if Coin == 0:       # EXPLOIT 
                    print('T=',T, 'Fois= ', fois, 't= ', t,'exploit')
                    H = update_H(w_edges)               # Update H according to current w_edges
                    chosen_path = exploit(H,w_edges)    # Draw a path by w_edges weights
                    loss = np.dot(bin_path(chosen_path), Loss(adver_stra) ) # A loss (scalar) generated by adversay
                    Cul_loss += loss
                    estimate = est_loss(chosen_path,w_edges,H,setting,loss)    # Unbiasly Estimate the loss according to setting
    #                print(estimate)
    
    #                eta=(lambda_min)* (t+1)**(-1/2)/(2*(B**2))  #Sasuake decreasing parameter
                      
                    w_edges = w_edges* np.exp( -eta *estimate )
            #        w_edges =  w_edges/(sum(w_edges)) 
    #                print(w_edges)
                elif Coin == 1:   # EXPLORE
                    print('T=',T, 'Fois= ', fois, 't= ', t,'explore')
                    H = update_H(w_edges)
                    chosen_path = explore()  ## Explore by either exploration='Uniform' or either ='Optimal'
                    loss = np.dot(bin_path(chosen_path), Loss(adver_stra) ) # A loss generated by adversay, unobserved by player
                    Cul_loss += loss
                    estimate = est_loss(chosen_path,w_edges,H,setting,loss)    # Unbiasly Estimate the loss according to setting
    #                print(estimate)
    
    #                eta=(lambda_min)* (t+1)**(-1/2)/(2*(B**2))   #Sasuake decreasing parameter
    
                    w_edges = w_edges* np.exp( -eta *estimate )
            #        w_edges =  w_edges/(sum(w_edges)) 
    #                print(w_edges)
            ###############################################################################
            ############    COMPUTE BEST ACTION IN HIND-SIGHT   ##########################
            K = np.zeros((m+1, n+1))
            V = np.zeros((m+1, n+1))
            #compute the value of best action to GAIN THE MAXIMUM
            for i in range(m+1):    # troops
                for j in range(1,n+1):      # battlefield
                    K[i,j] = blotto(i,adv_store[1,j-1])*values[j-1]
                    for t in range(2,T+1):
                        K[i,j] = K[i,j] +    blotto(i,adv_store[t,j-1])*values[j-1]  
                    temp = 0
                    for k in range(0,i+1):
                        temp = max(temp, V[k,j-1] + K[i-k,j])
                    V[i,j]= temp
                    
            regret = Cul_loss - (T-V[m,n])      #Cul loss minus the (min loss a.k.a. max gain)#
            print(regret)       
            #def test_optimize(a):
            #    test = 0
            #    for t in range(1,T+1):
            #        temp = blotto(a[0],adv_store[t,0])*values[0]
            #        for i in range(1,n):
            #            temp = temp + blotto(a[i],adv_store[t,i])*values[i]
            #        test = test + temp
            #    return test
            
            end = time.time()
            bound = (B**2)*np.log(P)/ gamma/lambda_min   +   gamma*lambda_min*(E/(B**2) + 2/lambda_min)*T
            ####################################################################################
            ######################         SAVING THE RESULTS        #########################
            #
            df = pd.read_csv(save, header =None)
            df1 =pd.DataFrame(np.append([n,m,p,T], [gamma_option, gamma, eta, exploration,lambda_min, setting, adv_distribution, end-start, T- V[m,n], regret, bound]))
            output= pd.concat([df,df1], ignore_index=False, axis=1)
            output.to_csv(save,header =None, index= False)