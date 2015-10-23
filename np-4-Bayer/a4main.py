
import sys
import numpy as np
import scipy.stats
import numpy.linalg as la

def  read():
    D = np.genfromtxt(sys.argv[1], dtype=None, delimiter=',')
    D1 = np.genfromtxt(sys.argv[1], delimiter=',')
    X = D1[:,:4]
    Y = [D[i][4] for i in range(len(D))]
    K = sys.argv[2]
    a = sys.argv[3]
    return (X,Y,K,a)

def bayes_params(X, Y):
    classes = np.unique(Y)
    len_x = len(X)
    len_c = len(classes)
    len_4 = len(X[0])
    Pc = np.zeros(len_c)
    mean = np.zeros([len_c, len_4])
    cov = np.zeros([len_c,len_4,len_4])
    for i in range(len_c):
        D = np.array([X[j] for j in range(len_x) if Y[j] == i]) # sorting
        len_d = len(D)
        Pc[i] = len_d/len_x
        mean[i] = np.mean(D, axis = 0)
        #center data
        Z = D - mean[i]
        cov[i] = np.dot(Z.T,Z)/len_d
        # to varifly 
        # E1 = np.cov(D.T, bias=1)
        # print(E1)
    return (classes, Pc, mean, cov)

def multi_var_normal_pdf(x, mean, cov):
    exp=np.exp(-np.dot(np.dot((x-mean).T,la.inv(cov)),(x-mean))/2)
    div = 1/((np.sqrt(np.pi*2)**len(x))*np.sqrt(la.norm(cov)))
    return div*exp


def paired_t_test(X, Y, K, alpha):
    n = len(X)
    K = int(float(K))
    alpha = float(alpha)
    diff = np.zeros(K)
    #list each class with number
    c = np.unique(Y)
    len_c = len(c)
    for i in range(n):
        for j in range(len_c):
            if(Y[i] == c[j]):
                Y[i] = j
                break

    for i_K in range(K):       
        # 1 compute training set X_i using bootstrap resampling
        sample = np.random.randint(n,size = n)
        X_i = X[sample]
        len_t = len(X_i)
         ## trying to do the same thing as "Y_i = Y[sample]", don't know why it failed
        Y_i = np.empty(len_t)
        for i in range(len_t):
            Y_i[i] = Y[sample[i]]
      
        # 2 train both full and naive Bayes on sample X_i
        classes, Pc, mean, cov = bayes_params(X_i, Y_i)

        # 3 compute testing set X - X_i
        #test on remaining ones
        rsample = np.setdiff1d(np.arange(n), sample)
        rX = X[rsample]
        len_r = n-len(np.unique(sample))
        # print(len_r, "==", len(rX))

        rY = np.empty(len_r)
        for i in range(len_r):
            rY[i] = Y[rsample[i]]
       
        # 4 assess both on X - X_i      
        rY_f = np.zeros(len_r)
        rY_n = np.zeros(len_r)
        for j in range(len_r): 
            p = np.zeros(len_c)#for full
            for i in range(len_c):
                p[i] = multi_var_normal_pdf(rX[j], mean[i], cov[i])*Pc[i]
            
            rY_f[j] = np.argmax(p)

        
        for j in range(len_r):
            p1 = np.zeros(len_c)#for naive
            for i in range(len_c):
                cov_naive = np.diag(cov[i])*np.identity(len(cov)+1) 
                p1[i] = multi_var_normal_pdf(rX[j], mean[i], cov_naive)*Pc[i]
            
            rY_n[j] = np.argmax(p1)

        bool_f = np.equal(rY, rY_f)
        bool_n = np.equal(rY, rY_n)
        num_err_full = len_r-np.sum(bool_f)
        num_err_naive = len_r-np.sum(bool_n)
        print('sample, full, naive:', i_K, num_err_full, num_err_naive)

        # 5 compute difference in error rates
        err_rate_full = num_err_full/len_r
        err_rate_naive = num_err_naive/len_r
        diff[i_K] = err_rate_full - err_rate_naive

    print('all differences:'); print(diff)
 
    # compute mean, variance, and z-score
    mean = np.mean(diff)
    var = np.mean((diff-mean)**2)
    z_score = (np.sqrt(K)*mean)/np.sqrt(var)
    print('z-score:', z_score)

    # compute interval bound using inverse survival function of t distribution
    bound = scipy.stats.t.isf((1-alpha)/2.0, K-1) 
    print('bound:', bound)

    # output conclusion based on tests
    if(-bound<z_score and z_score<bound):
        print('accept: classifiers have similar performance')
    else:
        print('reject: classifiers have significantly different performance')


# read in data and command-line arguments, and compute X and Y
X,Y,K,alpha= read()
paired_t_test(X, Y, K, alpha)