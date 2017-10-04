import numpy as np
from LocalPCA_v0 import *

def RMMEDA_operator(PopDec,K,M,XLow,XUpp):
    N,D  = PopDec.shape
    ## Modeling
    Model,probability = LocalPCA(PopDec,M,K)
    ## Reproduction
    OffspringDec = np.zeros((N,D))
    # Generate new trial solutions one by one
    for i in np.arange(N):
        # Select one cluster by Roulette-wheel selection
        k = (np.where(np.random.rand()<=probability))[0][0]
        # Generate one offspring
        if not len(Model[k]['eVector'])==0:
            lower = Model[k]['a'] - 0.25*(Model[k]['b']-Model[k]['a'])
            upper = Model[k]['b'] + 0.25*(Model[k]['b']-Model[k]['a'])
            trial = np.random.uniform(0,1)*(upper-lower) + lower # ,(1,M-1)
            sigma = np.sum(np.abs(Model[k]['eValue'][M-1:D]))/(D-M+1)
            OffspringDec[i,:] = Model[k]['mean'] + trial*Model[k]['eVector'][:,:M-1].conj().transpose() + np.random.randn(D)*np.sqrt(sigma)
        else:
            OffspringDec[i,:] = Model[k]['mean'] + np.random.randn(D)
        NN,D = OffspringDec.shape
        low = np.tile(XLow,(NN,1))
        upp = np.tile(XUpp,(NN,1))
        lbnd    = OffspringDec <= low
        ubnd    = OffspringDec >= upp
       # print OffspringDec
       # print lbnd
       # print ubnd
       # input()
        OffspringDec[lbnd] = 0.5*(PopDec[lbnd] + low[lbnd]); 
        OffspringDec[ubnd] = 0.5*(PopDec[ubnd] + upp[ubnd]);

            
    return OffspringDec
    
#RMMEDA_operator(PopDec,K,M)
