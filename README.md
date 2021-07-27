# Maximum Likelihood Estimators ERGMs

This program takes an input for the number of nodes in a graph, randomly draws parameter values from a uniform distrubtion on [-5,5], then calculates the probability of each graph occuring, it then plots the graphs and colors them by min max or neither. A graph is said to be a min if every graph within one edge toggle of itself has a higher probability. The same follows for max. The end of the program implements 2 linear regression models to test for association between the amount of mins and maxs and the difference in theta values.
