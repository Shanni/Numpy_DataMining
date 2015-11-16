to run the program: python3.4 iris.txt k

Expectation-maximization clustering and evaluation

This assignment is on clustering using the expectation-maximization algorithm and evaluation using the purity score.

Consider dataset iris.txt. It has 4 numeric dimensions and and a fifth column indicating the class.

Implement Algorithm 13.3, Expectation-Maximization (EM) Algorithm, in Chapter 13, using all but the last attribute, with the following specifics:

(1) For initialization, instead of setting up means, covariance matrices, and prior probabilities using lines 2-4 of Algorithm 13.3, do the following:

Assign the first n/k points to cluster 1, the second n/k points to cluster 2, and so on, and estimate the means, covariance matrices, and prior probabilities using lines 9-12 of Algorithm 13.3. If n is not a multiple of k, let later clusters have one fewer point, e.g., if n = 11 and k = 3, then the first two clusters have 4 points each, and the last one has 3 points.
(2) For each cluster, estimate the full covariance matrix, instead of a diagonal one.

(3) For termination, use ε = 0.001.

After running the EM algorithm, assign each data point xj to the cluster Ci that yields the highest probability P(Ci|xj) among all clusters.