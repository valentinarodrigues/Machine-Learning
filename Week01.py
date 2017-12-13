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


    








