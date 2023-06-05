import numpy as np


def calc_scatters(K):
    """
    Calculate scatter matrix:
    scatters[i,j] = scatter of the sequence with starting frame i and ending frame j 
    """
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n + 1, n + 1))
    # TODO: use the fact that K - symmetric
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)

    diagK2 = np.diag(K2)

    i = np.arange(n).reshape((-1, 1))
    j = np.arange(n).reshape((1, -1))
    print("calculating scatters matrix...")
    scatters = (
            K1[1:].reshape((1, -1)) - K1[:-1].reshape((-1, 1)) -
            (diagK2[1:].reshape((1, -1)) + diagK2[:-1].reshape((-1, 1)) -
             K2[1:, :-1].T - K2[:-1, 1:]) /
            ((j - i + 1).astype(np.float32) + (j == i - 1).astype(np.float32))
    )
    scatters[j < i] = 0
    
    
    
    ### same result. but, slower than above. for intuition ###
    # scatters2 = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(i,n):
    #         scatters2[i,j] = K1[j+1]-K1[i] - (K2[j+1,j+1]+K2[i,i]-K2[j+1,i]-K2[i,j+1])/(j-i+1)
    ###################################
    
    
    ### weave ###
    # scatters = np.zeros((n, n));
    # code = r"""
    # for (int i = 0; i < n; i++) {
    #     for (int j = i; j < n; j++) {
    #         scatters(i,j) = K1(j+1)-K1(i) - (K2(j+1,j+1)+K2(i,i)-K2(j+1,i)-K2(i,j+1))/(j-i+1);
    #     }
    # }
    # """
    # weave.inline(code, ['K1','K2','scatters','n'], global_dict = \
    #     {'K1':K1, 'K2':K2, 'scatters':scatters, 'n':n}, type_converters=weave.converters.blitz)
    #######################
    
    return scatters

def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=False, out_scatters=None):
    """Change point detection with dynamic programming
    arguments:
        K       - kernel between each pair of frames in video(square kernel matrix)
        ncp     - maximum number of change points(ncp>=0)
    Optional arguments:
        lmin     - minimum segment length
        lmax     - maximum segment length
        backtrack - when False - only evaluate objective scores (to save memory)
        verbose  - when True - print verbose message
        out_scatters - Output scatters
    
    Returns: (cps, obj)
        cps - detected array of change points: mean is thought to be constant on [ cps[i], cps[i+1] )    
        obj_vals - values of the objective function for 0..m changepoints
    """
    m = int(ncp) # prevent numpy.int64
    
    n, n1 = K.shape
    assert(n == n1), "Kernel matrix awaited."    
    
    assert((m + 1)*lmax >= n >= (m + 1)*lmin), ""
    assert(lmax >= lmin >= 1)
    
    J = calc_scatters(K)
    
    if out_scatters != None:
        out_scatters[0]=J
        
    # I[k, l] : value of the objective for k change-points and l first frames
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]
    
    if backtrack:
        # p[k, l] : "previous change". best t[k] when t[k+1] equals 1
        p = np.zeros((m+1, n+1), dtype=int)
    else:
        p = np.zeros((1,1), dtype=int)
    
    
    # method 1.######################################
    for k in range(1, m+1):
        for l in range((k+1) * lmin, n+1):
            I[k, l] = 1e100
            for t in range(max(k * lmin, l - lmax), l - lmin + 1):
                c = I[k-1, t] + J[t, l-1]
                if c < I[k, l]:
                    I[k, l] = c
                    if backtrack:
                        p[k, l] = t
    ##################################################
    
    # method 2.#######################################
    # for k in range(1, m+1):
    #     for l in range( (k+1)*lmin, n+1):
    #         tmin = max(k * lmin, l - lmax)
    #         tmax = l - lmin + 1
    #         c = J[tmin:tmax, l - 1].reshape(-1) + \
    #             I[k - 1, tmin:tmax].reshape(-1)
    #         I[k, l] = np.min(c)
    #         if backtrack:
    #             p[k, l] = np.argmin(c) + tmin
    ###################################################
    
    # Collect change points
    cps = np.zeros(m, dtype=int)
    
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]

    scores = I[:, n].copy() 
    scores[scores > 1e99] = np.inf
    return cps, scores

def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Main interface
    
    Detect change points automatically selecting their number
        K       - kernel between each pair of frames in video
        ncp     - number of change points to detect (ncp >= 0)
        vmax    - special parameter(??)
    Optional arguments:
        lmin     - minimum segment length
        lmax     - maximum segment length
        desc_rate - rate of descriptor sampling (vmax always corresponds to 1x)
    Note:
        - cps are always calculated in subsampled coordinates irrespective to
            desc_rate
        - lmin and m should be in agreement
    ---
    Returns: (cps, costs)
        cps   - best selected change-points
        costs - costs for 0,1,2,...,m change-points
        
    Memory requirement: ~ (3*N*N + N*ncp)*4 bytes ~= 16 * N^2 bytes
    That is 1,6 Gb for the N=10000.
    """
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)
    
    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling
    
    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)
    
    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)

    return (cps, costs)

def gen_data(n, m, d=1):
    """Generates data with change points
    n - number of samples
    m - number of change-points
    WARN: sigma is proportional to m
    Returns:
        X - data array (n X d)
        cps - change-points array, including 0 and n"""
    np.random.seed(1)
    # Select changes at some distance from the boundaries
    # cps = np.random.permutation((n*3//4)-1)[0:m] + 1 + n//8
    cps = np.random.permutation(n)[:m]
    
    cps = np.sort(cps)
    cps = [0] + list(cps) + [n]
    mus = np.random.rand(m+1, d)*(m/2)  # make sigma = m/2
    # mus = np.random.rand(m+1, d)  # make sigma = m/2
    X = np.zeros((n, d))
    for k in range(m+1):
        X[cps[k]:cps[k+1], :] = mus[k, :][np.newaxis, :] + np.random.rand(cps[k+1]-cps[k], d)
    return (X, np.array(cps))


if __name__ == '__main__':
    n = 1000 # number of samples
    m = 20 # number of maximum change-points
    d = 3 # dimension of data
    X, cps_gt = gen_data(n,m,d=d)
    K = np.dot(X, X.T) # make Gram matrix A
    cps, socres = cpd_auto(K, m, 1)