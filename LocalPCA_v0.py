import numpy as np


#PopDec = np.random.uniform(0,1,(10,3))
#PopDec = np.array([[0.341131  , 0.404572 ,  0.794194],
#                   [0.236895  , 0.794194 ,  0.111832],
#                   [0.760747  , 0.892365 ,  0.634125],
#                   [0.715132 ,  0.689792 ,  0.062184],
#                   [0.681623 ,  0.495598 ,  0.430207]])

#M = 2
#K = 3
#PopDec = np.array([[ 0.11652626,  0.89357241,  0.70015829,  0.63773254,  0.2080351 ,
#         0.14524755,  0.64224776,  0.95500061,  0.47148253,  0.70671527],
#       [ 0.01395715,  0.72884366,  0.0578533 ,  0.6300882 ,  0.00587139,
#         0.50596099,  0.83556115,  0.81412542,  0.60192359,  0.35937552],
#       [ 0.25637576,  0.1806496 ,  0.04905008,  0.64220236,  0.26139777,
#         0.72474985,  0.67723492,  0.38972384,  0.35364639,  0.2412986 ],
#       [ 0.53487242,  0.85238473,  0.54881505,  0.21869577,  0.14844504,
#         0.41930144,  0.82430799,  0.80312392,  0.32335593,  0.65464765],
#       [ 0.70061774,  0.78276963,  0.5850003 ,  0.52333649,  0.89203156,
#         0.83304142,  0.28598522,  0.11909428,  0.7601419 ,  0.05155682],
#       [ 0.76721705,  0.34016134,  0.52993109,  0.56924876,  0.82306252,
#         0.85969681,  0.28973394,  0.04106311,  0.55265343,  0.07952384],
#       [ 0.71931518,  0.77209922,  0.6967647 ,  0.37432725,  0.06976076,
#         0.39778583,  0.91116021,  0.90664652,  0.61589932,  0.33979642],
#       [ 0.81665224,  0.50868978,  0.86396178,  0.26707376,  0.52408515,
#         0.95971759,  0.69472365,  0.97385593,  0.03063698,  0.05134518],
#       [ 0.5880757 ,  0.09495876,  0.40362619,  0.79073955,  0.05536915,
#         0.87522403,  0.71448675,  0.09827789,  0.5836298 ,  0.12128568],
#       [ 0.40601575,  0.74199947,  0.82150987,  0.41057631,  0.21355527,
#         0.39867235,  0.64337383,  0.90130089,  0.07548961,  0.94617193]])
M = 2
K = 5

def LocalPCA(PopDec, M, K):
    N,D = np.shape(PopDec) # Dimensions
    Model = [     {'mean'   : PopDec[k],  # The mean of the model
                   'PI'     : np.eye(D),                  # The matrix PI
                   'eVector': [],                         # The eigenvectors
                   'eValue' : [],                         # The eigenvalues
                   'a'      : [],                         # The lower bound of the projections
                   'b'      : []} for k in range(K)]                           # The upper bound of the projections
    
    ## Modeling
    for iteration in range(1 , 50):
        # Calculte the distance between each solution and its projection in
        # affine principal subspace of each cluster
        distance = np.zeros((N,K)) # matrix of zeros N*K
        for k in range(K):
            distance[:,k] = np.sum((PopDec-np.tile(Model[k]['mean'],(N,1))).dot(Model[k]['PI'])*(PopDec-np.tile(Model[k]['mean'],(N,1))),1)
        # Partition
        partition = np.argmin(distance,1) # get the index of mins
        # Update the model of each cluster
        updated = np.zeros(K, dtype=bool) # array of k false
        for k in range(K):
            oldMean = Model[k]['mean']
            current = partition == k
            if sum(current) < 2:
                if not any(current):
                    current = [np.random.randint(N)]
                Model[k]['mean']    = PopDec[current,:]
                Model[k]['PI']      = np.eye(D)
                Model[k]['eVector'] = []
                Model[k]['eValue']  = []
            else:               
                Model[k]['mean']    = np.mean(PopDec[current,:],0)
                cc = np.cov( (PopDec[current,:] - np.tile(Model[k]['mean'],(np.sum(current),1))).T ) 
                eValue, eVector     = np.linalg.eig( cc )
                rank           = np.argsort(-(eValue),axis=0)
                eValue         = -np.sort(-(eValue),axis=0)
                Model[k]['eValue']  = np.real(eValue).copy()
                Model[k]['eVector'] = np.real(eVector[:,rank]).copy()
                Model[k]['PI']      = Model[k]['eVector'][:,(M-1):].dot(Model[k]['eVector'][:,(M-1):].conj().transpose())
                
            updated[k] = not any(current) or np.sqrt(np.sum((oldMean-Model[k]['mean'])**2)) > 1e-5

        # Break if no change is made
        if not any(updated):
            break

	## Calculate the smallest hyper-rectangle of each model
    for k in range(K):
        if len(Model[k]['eVector']) != 0:
            hyperRectangle = (PopDec[partition==k,:]-np.tile(Model[k]['mean'],(sum(partition==k),1))).dot(Model[k]['eVector'][:,0:M-1])
            Model[k]['a']     = np.min(hyperRectangle) # this should by tested
            Model[k]['b']     = np.max(hyperRectangle) # this should by tested
        else:
            Model[k]['a'] = np.zeros((1,M-1))
            Model[k]['b'] = np.zeros((1,M-1))
 
    
    ## Calculate the probability of each cluster for reproduction
    # Calculate the volume of each cluster
    volume = np.array([Model[k]['b'] for k in range(K)])-np.array([Model[k]['a'] for k in range(K)]) # this should be tested
#    volume = prod(cat(1,Model.b)-cat(1,Model.a),2)
    # Calculate the cumulative probability of each cluster
    probability = np.cumsum(volume/np.sum(volume))

    

    return Model,probability

