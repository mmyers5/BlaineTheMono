from __future__ import division
from scipy.integrate import nquad
import numpy as np
import os
import scipy.interpolate as interpol
import sklearn.neighbors
import struct

#if sigma is too big, lVec sucks ass and p(L) is never anything
# make h smaller, p(MTL) is too jagged
def main_func(xVeci, x1Veci, dimVec, dim1Vec, sims, h, sigma):
    
    # pre-tabulate p(L). Takes a while.    
    lVec = x1Veci + 5*np.outer(np.linspace(-1,1,10), sigma)                     # creates array of luminosities within 5*sigma_L of x1Veci
    lVec = np.hsplit(lVec, len(dim1Vec))                                        # split into separate arrays for cartesian product
    lVec = cartesian(lVec, out = None)                                          # input luminosities. shape = (points,dims)

    pList = l_func(lVec, dimVec, dim1Vec, sims,h, sigma)                        # p(L) that correspond to sets in lVec
    splFunc = interpol.LinearNDInterpolator(lVec, pList)

    #make a fit for kernel

    kde = sklearn.neighbors.kde.KernelDensity(kernel = 'epanechnikov', bandwidth = h)
    kde.fit(sims)
    return kde
    # reduce list of sims by removing outlier L
    limVec = np.array([x1Veci-5*sigma , x1Veci+5*sigma])
    sims = sims[~np.less(sims[:,dim1Vec],limVec[0]).any(axis=-1)]
    sims = sims[~np.greater(sims[:,dim1Vec],limVec[1]).any(axis=-1)]

    # call integral
    nonNormed = int_func(xVeci, x1Veci, dimVec, dim1Vec, sims, h, sigma, splFunc, kde)
    
    return nonNormed
    lower = np.apply_along_axis(min, 0, sims[:,dimVec])
    upper = np.apply_along_axis(max, 0, sims[:,dimVec])
    ranges = np.array([lower, upper]).T

    normVal = nquad(int_func, ranges, (x1Veci, dimVec, dim1Vec, sims, h, sigma, splFunc))[0]

    result = nonNormed/normVal

    return result

def int_func(*args):

    xVec = np.asarray(args[:-8])                                                # could get passed in for normalization, don't want a tuple
    x1Veci = args[-8]
    dimVec = args[-7]
    dim1Vec = args[-6]
    sims = args[-5]
    h = args[-4]
    sigma = args[-3]
    splFunc = args[-2]
    kde = args[-1]
    result = 0.

    # reduce list of sims by removing outlier M and T
    limVec = np.array([xVec-h , xVec+h])
    sims = sims[~np.less(sims[:,dimVec],limVec[0]).any(axis=-1)]
    sims = sims[~np.greater(sims[:,dimVec],limVec[1]).any(axis=-1)]
    if np.shape(sims)[0] == 0:
        return 0

    for sim in sims:
        x1Vec = sim[dim1Vec]
        ranges = ranger(x1Vec, h)
        result += nquad(integrand, ranges, (xVec, x1Veci, sigma, splFunc, kde))[0]
        print result
    return result
  
def integrand(*args):
    x1Vec = args[:-5]
    xVec = args[-5][-1] # -1 necessary or else it has an additional dimension
    x1Veci = args[-4]
    sigma = args[-3]
    splFunc = args[-2]
    kde = args[-1]
    f = open('test.txt', 'a')
    
    # p(MTL)
    K = np.exp(kde.score_samples(np.hstack([xVec, x1Vec])))
    
    # p(L|data)
    eFunc = np.sum( ((x1Vec - x1Veci)/(sigma*5))**2 )
    gFunc = np.exp( (-.5)*eFunc )

    # p(L)
    pL = splFunc(x1Vec)
    
    result = (K/pL)*gFunc if pL !=0 else 0
    f.write(str(K) + ': ' + str(gFunc) + ': ' + str(pL) + ': ' + str(result) + '\n')
    return result

def l_func(x1Vec, dimVec, dim1Vec, sims, h, sigma):
    
    numV = np.shape(dimVec)[0]                                                  # number of variables (M, T, ... )
    numL = np.shape(x1Vec)[0]                                                   # number of sample points
    filtNum = np.shape(x1Vec)[1]                                                # number of luminosity filters (L1, L2, ... )
    zL2 = np.zeros((numL, filtNum))                                             # holds distance information

    if numL > 1:

        chSize = int(1e8/(np.shape(sims)[0]))                                   # iterate over smaller pieces of lVec
        chSize = 10
        chList = np.arange(numL)
        chList = chList[chList % chSize == 0]                                   # list to iterate over every piece
        for i, col in enumerate(dim1Vec): # for ever luminosity filter
            prev = 0 # previous value set to 0
            for ch in chList[1:]: # for every chunk specified by chList
                if ch != chList[-1]: # as long as number isn't last
                    tmp = np.subtract.outer(sims[:,col], x1Vec[prev:ch,i])      # subtract every simulation from every luminosity in chunks                                          
                    tmp = np.sum(tmp**2, 0)                                     
                    zL2[prev:ch,i] = tmp                  
                    prev = ch
                    continue                  
                else:
                    tmp = np.subtract.outer(sims[:,col], x1Vec[prev:,i])        # for last chunk only
                    tmp = np.sum(tmp**2, 0)
                    zL2[prev:,i] = tmp
        zL2 = np.sum(zL2,1)                                                     # sum across row
        zL2[zL2 >= h**2] = np.nan
        a2 = zL2/h**2

    else:
        zL2 = np.sum( (sims[:,dim1Vec] - x1Vec)**2 )
        a2 = zL2/h**2 if zL2<h**2 else 0

    if numV == 0:
        p_L = (4/3) * (1 - a2)**(3/2) * h**numV 
    elif numV == 2:
        p_L = .5 * np.pi*(a2 - 1)**2*h**numV
    elif numV == 3:
        p_L = (8 * np.pi / 15) * (1 - a2)**(5/2) * h**numV
    elif numV == 4:
        p_L = -numV*2*(np.pi**2 / 6) * (a2 - 1)**3 * h**numV
    elif numV == 5:
        p_L = (16 * np.pi**2 / 105) * (1 - a2)**(7/2) * h**numV
    elif numV == 6:
        p_L = (np.pi**3 / 24) * (a2 - 1)**4 * h**numV
    elif numV == 7:
        p_L = (32 * np.pi**3 / 945) * (1 - a2)**(9/2) * h**numV
    elif numV == 8:
        p_L = -(np.pi**4 / 120) * (a2 - 1)**5 * h**numV
    elif numV == 9:
        p_L = (64 * np.pi**4 / 10395) * (1 - a2)**(11/2) * h**numV
    else:
        raise Exception, "DON'T PANIC."
    
    p_L[np.isnan(p_L)] = 0
    return p_L

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def ranger(varSims, width):
    """
    Creates a list of ranges for functions with nquad

    Parameters
    ----------
    varSims: array or float
        contains points corresponding to variables over which nquad
        would integrate
    width: float
        distance from varSims for limits of integration
    
    Returns
    ----------
    ranges: list
        list of ranges
    """
    numV = len(varSims)
    ranges = np.zeros(shape = (numV, 2))
    
    for i in range(numV):
            ranges[i,0], ranges[i,1] = varSims[i]-width, varSims[i]+width
    return ranges

def dat_data(filtNum = 2, datBase = 'out_dat.dat', dir = None):
    """
    Makes array of information grabbed from database
    
    Parameters
    ----------
    filtnum: int
        number of filters applied in database. Can be given by read_clusterphot
        in nFlux. Could probably come up with a better way...
    datBase:
        filename of database file
    dir: string
        if dir != None, then the file is searched for in the indicated
        directory. Otherwise the current working directory is used

    Returns
    ----------
    sims: array
        cluster information for each dimension. Every row is a new simulation,
        every column is a new dimension
    """

    if dir is not None:
        datBase = os.path.join(dir, datBase)
        
    dF = open(datBase, 'rb')                                                    
    dSize = os.path.getsize(datBase)
    dNentry = 5 + filtNum                                                       # number of entries of cluster information
    dLentry = (dSize // (8 * dNentry))                                          # number of lines of cluster information (or rows, simulations, etc...)
    #dLentry = 1000 # for testing
    
    # initializing vectors
    ageVec = np.zeros( (dLentry, 1) , 'd')                        
    massVec = np.zeros( (dLentry, 1) , 'd')
    lumVec = np.zeros( (dLentry, filtNum) , 'd')    
    
    # Grabbings logarithms of M, T, L1, L2, ..., Ln
    for i in range(dLentry):                                                    
        dRaw = dF.read(8 * dNentry)                                             # grab offsets from SlugAPI, cluster ID and most massive star removed
        massVec[i] = np.log10(struct.unpack_from('d', dRaw, offset = 8))       
        ageVec[i] = np.log10(struct.unpack_from('d', dRaw, offset = 16))
        lumVec[i][:] = np.log10(struct.unpack_from('d'*filtNum, dRaw, offset = 40))
    dF.close()

    massVec[massVec<0], ageVec[massVec<0], lumVec[lumVec<0] = 0., 0., 0.        # some are too small to make the cut
    sims = np.hstack((massVec, ageVec, lumVec))                                 # rows = individual sims, columns = dimensions 
    
    return sims

def integrand_test(xVeci, x1Veci, dimVec, dim1Vec, sims, h, sigma, splFunc):
     
     # p(MTL)
    index = np.shape(sims)[-1]
    sims = np.c_[sims, sims[:,dim1Vec]]                                         # new columns for editing in-place
    tmp = ( (sims[:,dimVec] - xVeci)/h )**2
    mtlArray = np.sum(1 - tmp, 1)

    sims[:,index:] = np.einsum('ij,i->ij', sims[:,index:], mtlArray)
    sims = np.delete(sims, np.where( np.any(sims <= 0 , 1)), 0)
    sims[:,index] = np.sum(sims[:, index:], 1)
    sims = sims[:,:-1]

    # p(L|data)
    tmp = ( (x1Veci - sims[:,dim1Vec])/sigma )**2
    tmp = np.sum(tmp, 1)
    sims[:,-1] = sims[:,-1] * np.exp(-0.5*tmp)
    sims = np.delete(sims, np.where( np.any(sims <= 0 , 1)), 0)

    integrand = sims[:,-1]
    pL = splFunc(sims[:,dim1Vec])
    integrand = np.c_[sims[:,dim1Vec],integrand, pL]
    integrand = np.delete(integrand, np.where( np.any( integrand == np.inf, 1)), 0)
    integrand[:,-2] = integrand[:,-2] / integrand[:,-1]
    integrand = integrand[:,:-1]
    
    return integrand