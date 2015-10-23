##Numeric Data 
This program is by Shan Liu.
To test the code, type python3.4 
run with python3.4 a1main.py magic_04.data

#GOAL of this program:
	Manipulate and get statistical facts from data.
	e.g. Multivariate mean vector; Covariance matrix; Dominant eigenvalue and eigenvector of covariance matrix…

#DETAILS:
1. Compute the multivariate mean vector.

2. Compute the sample covariance matrix as inner products between the columns of the centered data matrix (see equation 2.30 in Chapter 2). Verify that your answer matches the one using the numpy cov function (using bias=1).

3. Compute the attribute that has the largest and smallest variance, respectively.

4. Compute the pair of attributes that has the largest and smallest covariance, respectively.

5. Compute the dominant eigenvalue and eigenvector of the covariance matrix via the power iteration method (on pages 105-106 in Chapter 4), using the threshold value of 0.0001 for ε. Verify your answer using the numpy linalg.eig function.