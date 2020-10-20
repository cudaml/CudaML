import torch



class LinearRegression():
    '''
    OLS Regression
    '''
    def __init__(self,fit_intercept=True,normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
    
    def fit(self, X, y, sample_weight=None):
        if torch.cuda.device_count()<1:
            raise RuntimeError('GPU NOT FOUND')
            
        X = torch.tensor(X).cuda()
        y = torch.tensor(y).cuda()
        
        if len(X.shape)==1: X = self._reshapeX(X)
            
        ones = torch.ones(X.shape[0]).view(-1,1).cuda()
        X = torch.cat((ones,X),1)
        
        ## Coefficients
        self.coeffiecients = ((torch.inverse(X.T@X)) @ X.T) @ y
        
    def predict()
        pass
        
    def _reshapeX(X):
        return X.view(-1,1)