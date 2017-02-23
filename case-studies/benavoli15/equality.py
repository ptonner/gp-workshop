import GPy, scipy
import numpy as np

def test(x,y1,y2):

    n = x.shape[0]

    xgp = np.zeros((n*2, 2))
    xgp[:,0] = np.tile(x,2)
    xgp[n:,1] = 1

    # input must be 2 dimensional
    ygp = np.concatenate((y1,y2))[:,None]

    kern = GPy.kern.IndependentOutputs(GPy.kern.RBF(1))
    m = GPy.models.GPRegression(xgp,ygp, kern)
    m.randomize()
    m.optimize()

    xpred = np.zeros((n*2,2))
    xpred[:,0] = np.tile(np.sort(x),2)
    xpred[n:,1] = 1

    mu,cov = m.predict_noiseless(xpred,full_cov=True)

    op = np.zeros((n,n*2))
    for i in range(n):
        op[i,i] = 1
        op[i,i+n] = -1

    mu = np.dot(op,mu)
    cov = np.dot(op, np.dot(cov, op.T))
    std = np.sqrt(np.diagonal(cov))

    evals = np.linalg.eigvals(cov)
    nu = sum(evals > 1e-3)

    alpha = scipy.stats.chi2.cdf(np.dot(mu[:,0], np.dot(cov, mu[:,0])), nu)

    return alpha
