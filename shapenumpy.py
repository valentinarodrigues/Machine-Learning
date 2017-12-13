import numpy as np
#X = np.array([[1, 2]])

X = np.array([[1, 2],[5, 6],[9,10]])
#print(X.shape)
#print(np.zeros((X.shape[0], 6)))
Xnew=np.zeros((X.shape[0], 6))
#print(X[:,1])
#print(X[0,0],X[0,1])
for i in range(X.shape[0]):
    Xnew[i][0]=X[i][0]
    Xnew[i][1]=X[i][1]
    Xnew[i][2]=X[i][0]*X[i][0]
    Xnew[i][3]=X[i][1]*X[i][1]
    Xnew[i][4]=X[i][0]*X[i][1]
    Xnew[i][5]=1
print(Xnew)
w= np.linspace(-1, 1, 6)
wx=np.zeros((6))

wx=Xnew.dot(w)
# for coursera wx=X_expanded=expand(X)
p=np.zeros((6))
for i,wx1 in enumerate(wx):
    p[i]=1/(1+np.exp(-wx1))
print(p)



"""
>>> X.shape
(4,)
>>> y = np.zeros((2, 3, 4))
>>> y.shape
(2, 3, 4)
>>> y.shape = (3, 8)
>>> y
array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
>>> y.shape = (3, 6)

for i in range(X.shape[0]):
    Xn[i][0]=X[i][0]
    Xn[i][1]=X[i][1]
    Xn[i][2]=X[i][0]*X[i][0]
    Xn[i][3]=X[i][1]*X[i][1]
    Xn[i][4]=X[i][0]*X[i][1]
    Xn[i][5]=1
print(Xnew)
"""
dummy_weights = np.linspace(-1, 1, 6)
print(dummy_weights)
