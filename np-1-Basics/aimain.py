import numpy as np
import sys

# data = open(sys.argv[1],'rU').read()

data=np.genfromtxt(sys.argv[1], delimiter=',')

print("Number of points: ", data.shape[0])
print("Number of dimensions: ", data.shape[1])

print("1. Multivariate mean vector:\n", data.mean(axis=0))
datan=data[:,:10]
a=datan-datan.mean(axis=0)
Covariance=np.dot(a.T,a)/data.shape[0]
print("2. Covariance matrix:\n",Covariance)
#valify...
Covariance1=np.cov(datan.T,bias=1)
flag="don't know"
if np.array_equal(Covariance1,Covariance)==True:
	flag="equal"
print("Verify using numpy cov:", flag)


listAttr=["1. fLength: continuous # major axis of ellipse [mm]", 
"2. fWidth: continuous # minor axis of ellipse [mm] ",
"3. fSize: continuous # 10-log of sum of content of all pixels [in #phot] ",
"4. fConc: continuous # ratio of sum of two highest pixels over fSize [ratio]", 
"5. fConc1: continuous # ratio of highest pixel over fSize [ratio] ",
"6. fAsym: continuous # distance from highest pixel to center, projected onto major axis [mm]", 
"7. fM3Long: continuous # 3rd root of third moment along major axis [mm]", 
"8. fM3Trans: continuous # 3rd root of third moment along minor axis [mm] ",
"9. fAlpha: continuous # angle of major axis with vector to origin [deg] ",
"10. fDist: continuous # distance from origin to center of ellipse [mm] "]
#Q3 
dia=Covariance1.diagonal()
min=np.nanmin(dia)
mindex=np.argmin(dia)
max=np.nanmax(dia)
maxdex=np.argmax(dia)
print("3. Attribute ", listAttr[maxdex] ,"has largest variance ",max)
print("   Attribute ", listAttr[mindex] ,"has smallest variance ",min)

#Q4
#get lower tri-matrix of Coveriance1
ui1=np.triu_indices(Covariance1.shape[0])
Covariance1copy=Covariance1
Covariance1copy[ui1]=0
maxx=Covariance1copy.max()
minn=Covariance1copy.min()
maxxdex=np.argmax(np.ravel(Covariance1copy))
maxxdex1=divmod(maxxdex,Covariance1copy.shape[1])
minndex=np.argmin(np.ravel(Covariance1copy))
minndex1=divmod(minndex,Covariance1copy.shape[1])
print("4. Attribute pair (",listAttr[maxxdex1[0]], listAttr[maxxdex1[1]],") has largest covariance ", maxx) 
print("   Attribute pair (",listAttr[minndex1[0]], listAttr[minndex1[1]],") has smallest covariance ", minn)   

#Q5
from numpy import linalg as LA
np.seterr(divide='ignore', invalid='ignore')
print("5. Dominant eigenvalue and eigenvector of covariance matrix:")
#make a copy co in order to compute
co=np.cov(datan.T,bias=1)
p1=np.ones(co.shape[0])
p=np.zeros(co.shape[0])
e=.0001
eigenvalue=1
temp=0
temppi=0
while LA.norm(p-p1)>e:
	p=p1
	p1=np.dot(co.T,p1)
	i=np.argmax(p1)
	#print("#in while loop i:",i)
	#print(p1)
	temppi=p[i]
	temp=p1[i]
	eigenvalue=temp/temppi
	p1=p1/temp
    
	#print("#in while loop matrix is :",temp)
p1=p1/LA.norm(p1)
print (eigenvalue)
print (p1)

#print(eigenvalue)
#print(eigenvector)
w,v=LA.eig(co)
i=np.argmax(w)
flag1="don't know"
print("Verify using numpy linalg.eig:")
if (w[i]==eigenvalue and v[:,i]-p1):
	flag1="equal"
	print(flag1)
else:
	print("difference:")
	print(w[i]-eigenvalue)
	print(v[:,i]-p1)

