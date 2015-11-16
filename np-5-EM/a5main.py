import sys
import numpy as np 
import numpy.linalg as la
import decimal as de

def read():
	D = np.genfromtxt(sys.argv[1],delimiter=",")
	len_D = len(D)
	Dy = [np.genfromtxt(sys.argv[1], dtype=None, delimiter=',')[i][4] for i in range(len_D)]
	uDy = np.unique(Dy)
	len_uDy = len(uDy)
	for i in range(len_D):
		for j in range(len_uDy):
			if(Dy[i]==uDy[j]):
				Dy[i] = j
				break
	D = D[:,:4]
	k = sys.argv[2]
	return (D,k,Dy,uDy)

# def multi_var_normal_pdf(x, mean, cov):
#     exp=np.exp(-np.dot(np.dot((x-mean),la.inv(cov)),(x-mean).T)/2)\
#     /(np.power(np.sqrt(np.pi*2),len(x))*la.det(cov))
  
#     return exp

def multi_var_normal_pdf(x, mean, cov):
    d = len(x)
    dev = x - mean
    return (np.power(2 * np.pi, -d/2) *
            np.power(np.linalg.det(cov), -1/2) *
            np.exp(-1/2 * np.dot(np.dot(dev.T, np.linalg.inv(cov)), dev)))

def Expectation_Maximization(D,k,e,Dy,uDy):
	k = int(k)
	len_k = len(D)
	len_ck = int(len(D)/k)
	dim = len(D[0])
	mean, cov, p = np.empty([k,dim]),np.empty([k,dim,dim]),np.empty(k)
	for i in range(k):
		mean[i] = np.mean(D[i*len_ck:(i+1)*len_ck], axis=0)
		cov[i] = np.cov(D[i*len_ck:(i+1)*len_ck].T, bias=1)
		p[i] = 1/k

	mean_i, cov_i, p_i = np.empty([k,dim]),np.empty([k,dim,dim]),np.empty(k)
	w = np.empty([k,len_k])
	count=0
	while np.sum([la.norm(mean[i_w]-mean_i[i_w])**2 for i_w in range(k)]) > e :
		
		mean_i = np.copy(mean)
		#Expectation Step
		for i in range(k):
			for j in range(len_k):
				w[i][j] = multi_var_normal_pdf(D[j],mean[i],cov[i])*p[i]/\
				np.sum([multi_var_normal_pdf(D[j],mean[a],cov[a])*p[a] for a in range(k)])
	
		#Maximization Step
		for i in range(k):
			mean[i] = np.sum([w[i][j]*D[j] for j in range(len_k)], axis=0)/\
			np.sum([w[i][j] for j in range(len_k)])
			cov[i] = np.sum([w[i][j]*np.dot((D[j]-mean[i])[:,None],(D[j]-mean[i])[None]) for j in range(len_k)], axis=0)/\
			np.sum([w[i][j] for j in range(len_k)])
			p[i] = np.sum([w[i][j] for j in range(len_k)])/len_k
		count = count+1
		
	print('number of iterations ',count)
	assign = np.argmax(w,axis=0)
	for i in range(k):
		print('cluster ', i)
		print('size', np.sum([ai==i for ai in assign]))
		index = np.where(assign==i)
		print('points\n', index)
		print('mean\n',mean[i])
		print('cov\n',cov[i])
	k_val=np.zeros([k,k])
	for i in range(len_k):
		k_val[assign[i]][Dy[i]] += 1
	print(k_val)			
	print('purity', np.sum([np.max(k_val[:,i]) for i in range(k)]) /len_k)

	print('confusion matrix (this part is for bonus only)')
	for i in range(k):
		print(uDy[i],k_val[i])

D,k,Dy,uDy = read()
Expectation_Maximization(D, k , .001, Dy, uDy)


