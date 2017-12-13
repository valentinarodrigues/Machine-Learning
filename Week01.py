def expand(X):
    """
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature1*feature2, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))
    Xnew=np.zeros((X.shape[0], 6))
    for i in range(X.shape[0]):
        Xnew[i][0]=X[i][0]
        Xnew[i][1]=X[i][1]
        Xnew[i][2]=X[i][0]*X[i][0]
        Xnew[i][3]=X[i][1]*X[i][1]
        Xnew[i][4]=X[i][0]*X[i][1]
        Xnew[i][5]=1
    return Xnew
    # TODO:<your code here>
def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see description above
        
    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    
    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """
    X_expanded = expand(X)
    wx=np.zeros((6))
    wx=X_expanded.dot(w)
    p=np.zeros((wx.shape))
    for i,wx1 in enumerate(wx):
        p[i]=1/(1+np.exp(-wx1))
    return p
    # TODO:<your code here>


def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function using formula above.
    """
    pi1=np.zeros((y.shape))
    pi2=np.zeros((y.shape))
    pi1=probability(X,w)
    for j,t in enumerate(y):
        pi2[j]=np.multiply(t,(np.log(pi1[j]))) + np.multiply((1-t),(np.log(1-pi1[j])))
    l=-(np.sum(pi2))/y.shape
    return l
    # TODO:<your code here>

def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    """    
    i=np.dot((probability(X, w) - y),X) after manually solving derivative
    n=y.shape
    l=i/n #divide by number of samples
    return l
    
    # TODO<your code here>

from IPython import display

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()
# please use np.random.seed(42), eta=0.1, n_iter=100 and batch_size=4 for deterministic results

np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)
    w=w-(eta*compute_grad(X_expanded, y, w))#dont divide by number of samples since already divided
visualize(X, y, w, loss)
plt.clf()








