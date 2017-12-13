import numpy as np
def expand(X):
    Xnew=np.zeros((X.shape[0], 6))
    for i in range(X.shape[0]):
        Xnew[i][0]=X[i][0]
        Xnew[i][1]=X[i][1]
        Xnew[i][2]=X[i][0]*X[i][0]
        Xnew[i][3]=X[i][1]*X[i][1]
        Xnew[i][4]=X[i][0]*X[i][1]
        Xnew[i][5]=1
    return Xnew

X1 = np.array([[ 1.20798057,  0.0844994 ],[ 1.20798057,  0.0844994 ],
               [ 1.20798057,  0.0844994 ],[ 1.20798057,  0.0844994 ],
[ 1.20798057,  0.0844994 ],[ 1.20798057,  0.0844994 ],
 [ 0.76121787,  0.72510869], [ 0.76121787,  0.72510869],
 [ 0.55256189,  0.51937292]])
y=np.array([1,1,1,1,1,1,1,1,1])
w=np.linspace(-1, 1, 6)
X = expand(X1)

def compute_grad(w, X, y):
    def gradient(w, X, y):
        i=np.dot((nn(X, w) - y), X)
        return i

# Define the logistic function
    def logistic(z): 
        return 1 / (1 + np.exp(-z))

    def nn(x, w): 
        return logistic(x.dot(w.T))


    l=gradient(w, X, y)
    return l
print(compute_grad(w, X, y))

