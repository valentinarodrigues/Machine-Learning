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

def probability(X, w):
    X_expanded = expand(X)
    wx=np.zeros((6))
    wx=np.dot(X_expanded,w)
    p=np.zeros((wx.shape))
    for i,wx1 in enumerate(wx):
        p[i]=1/(1+np.exp(-wx1))
    return p


def compute_loss(X1, y, w):
    pi1=np.zeros((y.shape))
    pi2=np.zeros((y.shape))
    pi1=probability(X,w)
    for j,t in enumerate(y):
        pi2[j]=np.multiply(t,(np.log(pi1[j]))) + np.multiply((1-t),(np.log(1-pi1[j])))
    l=-(np.sum(pi2))/y.shape
    return l
#print(compute_loss(X1, y, w))


def compute_grad(X, y, w):
    i=np.dot(y-(probability(X, w)), X)/y.shape
    return i
print(compute_grad(X,y,w))

"""
def compute_grad(X, y, w):


    z = probability(X, w)
    t=z-y
    print(z-y)
    li=np.zeros(z.shape)
    lj=np.zeros(z.shape)
    for i,yi in enumerate(y):
        li[i]=z[i]-yi
    #l=np.dot(li,X)
    l=np.dot(t,X)   
    return l
    """
"""
    Ls=compute_loss(X1, y, w)
    L=np.zeros((6))
    e=np.diff(L,w)
    print(e)
    # TODO<your code here>

    
    m = X.shape[0]
    z = probability(X, w)
    total = np.sum(X.T*z-y)
    grad = (1/m) * total


    def compute_grad(X, y, w):
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.

    xTrans = X.transpose()
    m = X.shape[0]
    alpha = 0.05
    for i in range(0,100000):
        loss = np.dot(X, w) - y
        gradient = alpha * (1/m) * np.dot(xTrans, loss) 
        w = w-gradient
    return w
"""








