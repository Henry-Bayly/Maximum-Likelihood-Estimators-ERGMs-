##MLE Program
##Henry Bayly

##This program graphs the space of graphs for 'n' nodes and connects graphs with
##an edge if they can be reached by toggling an edge. the program further
##calculates max and min probabilities according to an ERGM model, and displays
## these graphs for a range of parameter values

import math
import numpy as np
import networkx as nx
import matplotlib 
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import random
from sklearn.linear_model import LinearRegression
##from mlxtend.plotting import plot_linear_regression
 

def create_graphs(N):
    a=[]
    for i in range(2**N):
        a.append(f'{bin(i)[2:]:0>{N}}')
    return a

def find_edges(graphs):
    edge_counts = []
    count = 0
    for x in graphs:
        for i in range(0,len(x)):
            if x[i] == '1':
                count += 1
            else:
                continue
        edge_counts.append(count)
        count=0
        
    return edge_counts


def make_adjacency(edge,n):
    matrix = [ [] for i in range(0,n)]
    for i in range(0,n):
        for x in range(0,n):
            matrix[i].append('x')
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = '0'
    count = 0
    matrix_count = 0
    x=1
    while count != n*(n-1)/2:
        for i in range(x,n):
            matrix[matrix_count][i] = edge[count]
            matrix[i][matrix_count] = edge[count]
            count += 1
        matrix_count += 1 
        x+=1
    return matrix


def matrix_cuber(matrix_cub):
    for i in range(0,len(matrix_cub)):
        for x in range(0,len(matrix_cub)):
            matrix_cub[x][i] = int(matrix_cub[x][i])
            
    cubed = np.linalg.matrix_power(matrix_cub, 3)
    for i in range(0,len(cubed)):
        for x in range(0,len(cubed)):
            cubed[x][i] = str(cubed[x][i])
    return cubed


def sum_diagonal(matrix_loc):
    count = 0
    for i in range(0,len(matrix_loc)):
        for x in range(0,len(matrix_loc)):
            matrix_loc[x][i] = int(matrix_loc[x][i])
            count = count + matrix_loc[x][i]
    return count

def find_triangles(graphs, n):
    triangles = []
    for x in graphs:
        matrix = make_adjacency(x,n)
        cubed_matrix = matrix_cuber(matrix)
        diagonal_total = sum_diagonal(cubed_matrix)
        num_triangles = diagonal_total / 6
        triangles.append(int(num_triangles))
    return triangles

def normal_constant(edge,triangles,theta_1,theta_2):
    count = 0
    for x in range(0,len(edge)):
        count += math.exp(edge[x]*theta_1+triangles[x]*theta_2)
    return count     



#c_norm is normalizing constant
def probabilities(x_obs,c_norm, triangles,theta_1,theta_2):
    #let theta_1 = 1 initially
    probs =[]
    for i in range(0,len(triangles)):
        probs.append(math.exp(theta_1*x_obs[i]+theta_2*triangles[i]) / c_norm)
    return probs

def toggle(i,edge):
    new_string = ''
    if edge[i]=='0':
        for x in range(0,len(edge)):
            if x==i:
                new_string=new_string+ '1'
            else:
                new_string=new_string+edge[x]
    else:
        for x in range(0,len(edge)):
            if x==i:
                new_string=new_string+ '0'
            else:
                new_string=new_string+edge[x]
    return new_string

def step_length(edges):
    #print("(a,b) means graphs 'a' and 'b' from list of graphs (starting at 1) are connected")
    pairs = []
    for x in range(0,len(edges)):
        for i in range(0,len(edges[x])):
            for y in range(0,len(edges)):
                hold = toggle(i,edges[x])
                if edges[x]==edges[y]:
                    continue
                if hold == edges[y]:
                    if (y,x) in pairs:
                        continue
                    else:
                        pairs.append((x,y))
                    
    return pairs


def is_min(graphs, probs,n):
    mins = []
    holder = -1
    for i in range(int(2**(n*(n-1)/2))):
        if holder != -1:
            mins.append(holder)
            holder = -1
        for j in range(int(2**(n*(n-1)/2))):
            if (i,j) in graphs or (j,i) in graphs:
                if probs[i] < probs[j]:
                    holder = i
            if (i,j) in graphs or (j,i) in graphs:
                if probs[i] >= probs[j]:
                    holder=-1
                    break
       
    if holder != -1:
        mins.append(holder)
    return mins

def is_max(graphs, probs,n):
    maxs = []
    holder = -1
    for i in range(int(2**(n*(n-1)/2))):
        if holder != -1:
            maxs.append(holder)
            holder = -1
        for j in range(int(2**(n*(n-1)/2))):
            if (i,j) in graphs or (j,i) in graphs:
                if probs[i] > probs[j]:
                    holder = i
            if (i,j) in graphs or (j,i) in graphs:
                if probs[i] <= probs[j]:
                    holder=-1
                    break
        
    if holder != -1:
        maxs.append(holder)
    return maxs
    
def eliminator(graphs,edges,triangles):
    new = []
    for x in range(len(graphs)):
        if (edges[x],triangles[x]) in new:
            continue
        else:
            new.append((edges[x],triangles[x]))
    return new

def grapher(graphs,edges,mins,maxs,n):
    Graph = nx.Graph()
    neither =[]
    for x in range(len(graphs)):
        if x not in mins and x not in maxs:
            neither.append(x)
        Graph.add_node(x)
    pos ={}
    count=0
    lables ={}
    for x in range(0,int(2**(n*(n-1)/2))):
        pos[count] = (random.randint(-20,20),random.randint(-20,20))
        count += 1
        lables[x] = x
    for x in range(0,len(edges)):
        for y in range(0,len(edges)):
            if (x,y) in edges:
                Graph.add_edge(x, y)
            else:
                continue
    nx.draw_networkx_nodes(Graph,pos,nodelist = mins, node_color='g')
    nx.draw_networkx_nodes(Graph,pos,nodelist = maxs, node_color='r')
    nx.draw_networkx_nodes(Graph,pos,nodelist = neither, node_color='b')
    nx.draw_networkx_edges(Graph,pos)
    nx.draw_networkx_labels(Graph,pos,lables)
    plt.show()
    
    


##Main Section
print("How many nodes are in graph? ")
N = int(input())
##print("What value would you like for theta_1, the edge weight? ")
##theta_one = float(input())
##print("What value would you like for theta_2, the triangle weight? ")
##theta_two = float(input())

theta_pairs = []
max_mins = []
theta_1s = []
theta_2s = []  
max_terms = []
min_terms = []
unique = []
graphs = create_graphs(int(N*(N-1)/2))
edges = find_edges(graphs)
triangles = find_triangles(graphs, int(N))
step_graphs = step_length(graphs)
for i in range(10):
    #theta_one = random.uniform(-5,5)
    theta_one = 2
    theta_1s.append(theta_one)
    #theta_two = random.uniform(-5,5)
    #print(theta_one, theta_two)
    theta_two = -3
    theta_2s.append(theta_two)
    theta_pairs.append((theta_one,theta_two))
    norm = normal_constant(edges,triangles,theta_one,theta_two)
    #print(norm, "\n")
    probs = probabilities(edges,norm, triangles,theta_one,theta_two)
    #print(1)
    mins = is_min(step_graphs,probs,N)
    #print(2)
    min_terms.append(mins)
    maxs = is_max(step_graphs,probs,N)
    #print(3)
    max_terms.append(maxs)
    max_mins.append((maxs,mins))
   # new = eliminator(graphs,edges,triangles)
    grapher(graphs,step_graphs,mins,maxs,N)
   # grapher(new,step_graphs,mins,maxs,N)

num_mins = []
num_maxs= []
for i in range(len(min_terms)):
    num_mins.append(len(min_terms[i]))
    num_maxs.append(len(max_terms[i]))
difference_theta = []
for i in range(len(theta_1s)):
    diff = abs(theta_1s[i] - theta_2s[i])
    difference_theta.append(diff)

##num_mins = np.array(num_mins)
##difference_theta = np.array(difference_theta).reshape((-1,1))
##model = LinearRegression()
##model.fit(difference_theta,num_mins)
##r_sq = model.score(difference_theta,num_mins)
##print("Regression for mins")
##print('coefficient of determination', r_sq)
##print('intercept:', model.intercept_)
##print('slope:', model.coef_ , '\n')
##
##x = np.linspace(0,8)
##y = np.linspace(0,3)
##plt.scatter(difference_theta,num_mins, color = 'black')
##plt.xlabel("Difference in Theta values")
##plt.ylabel("Number of Mins")
####plt.plot(x,y*model.coef_ + model.intercept_,'-r')
##plt.show()
##
##num_maxs = np.array(num_maxs)
##
##modeltwo = LinearRegression()
##modeltwo.fit(difference_theta,num_maxs)
##r_sq = modeltwo.score(difference_theta,num_maxs)
##print("Regression for maxs")
##print('coefficient of determination', r_sq)
##print('intercept:', modeltwo.intercept_)
##print('slope:', modeltwo.coef_)
##plt.scatter(difference_theta,num_maxs, color = 'black')
##plt.xlabel("Difference in Theta values")
##plt.ylabel("Number of Maxs")
##plt.show()
##is there a relationship between the difference in theta values and the amount of max/min values
##understand data types then run regression in python


                         
