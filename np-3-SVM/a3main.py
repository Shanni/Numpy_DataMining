import numpy as np
import sys
import numpy.linalg as la

D_0 = np.genfromtxt(sys.argv[1], delimiter=',')
dim = len(D_0)

D_1 = np.ones(dim)
D_1 = D_1.reshape(dim,1)
D = np.concatenate((D_0[:,:2], D_1), axis=1)

K = np.zeros([dim,dim])
for i in range(dim):
	for j in range(dim):
		K[i][j]=np.dot(D[i].T,D[j])

ita = np.zeros(dim)
for k in range(dim):
	ita[k]=1/K[k,k]

a = np.ones(dim)
a0 = np.zeros(dim)
y = D_0[:,2]

C = int(sys.argv[2])
epsilon = 0.0001
while la.norm(a - a0) > epsilon:
	a = np.copy(a0)
	for k in range(dim):
		# for i in range(dim):
			# sum[k]=sum[k]+a0[i]*y[i]*np.dot(K[i].T,K[k])
		sum = np.dot(a0*y, K[:,k]) # modified
		a0[k] = a0[k] + ita[k]*(1 - y[k]*sum)

		if(a0[k]<0):
			a0[k]=0
		if(a0[k]>C):
			a0[k]=C


boolean = np.zeros(dim)
for i in range(len(a0)):
	if(a0[i] != 0):
		boolean[i] = 1;


Data = np.zeros([2,dim])
Data[0] = D_0[:,0]*boolean
Data[1] = D_0[:,1]*boolean

w = np.dot(a0*y,Data.T)

b = y - np.dot(w, Data)
count = np.sum(boolean)

b_mean = np.sum(b) / count

print('Support vector indicies and alphas:')
for i in range(dim):
	if(boolean[i] == 1):
		print(i, " ", a0[i])
print('Number of support vectors: ', count)
print('Hyperplane weight vector and bias: ', w, b_mean)