import tensorflow as tf
import numpy as np
import warnings

def mvrnorm(mean, cov, n = 1, isChol = False, nugget = 1e-9):
    d = cov.shape[0]
    if isChol:
        L = cov
    else:
        L = np.linalg.cholesky(cov + nugget*np.identity(d))
    v = np.random.randn(d, n)
    x = np.matmul(L, v) + np.array([mean]*n).T
    return(x.T)
    
    
def fproj_list(F, nbterms = None, lv_inertia = .95):
    def tf_conv(x):
        x = x - tf.expand_dims(tf.reduce_mean(x, axis=1), 1)
        fact = tf.cast(tf.shape(x)[1] - 1, tf.float64)
        return tf.matmul(x, tf.math.conj(tf.transpose(x))) / fact
    
    nf = len(F) #nb of functional inputs
    nrep = F[0].shape[1] # nb of functional replicates
    
    # computing covariance for each functional input
    cov_list = [tf_conv(F_i) for F_i in F]
    
    # computing projection onto PCA basis functions
    eigDecomp = [tf.linalg.eigh(cov_i) for cov_i in cov_list]
    eigVals = [eig_i[0][::-1] for eig_i in eigDecomp]
    eigVects = [eig_i[1][:,::-1] for eig_i in eigDecomp]
    
    # truncating the PCA projection
    if nbterms is None:
        # using the inertia criteria
        nbterms = [np.min(np.where(np.cumsum(eigVals[i])/np.sum(eigVals[i]) >= lv_inertia))+1 for i in range(nf)]
    else:
        # using a predefined value
        if isinstance(nbterms, (int, float)):
            nbterms = np.repeat(int(nbterms), nf)
        elif len(nbterms) != nf:
            nbterms = np.repeat(int(nbterms[0]), nf)
            warnings.warn("'nbterms' should be an array sequence of len(F). "
                          "Only its first element will be considered instead.")
    
    # passing parameters of the projection to be used
    basis = [eigVects[i][:,:nbterms[i]] for i in range(nf)] # PCA basis functions
    coef = [tf.linalg.matmul(tf.transpose(basis[i]), F[i]) for i in range(nf)] # PCA coefficients
    coef = [[coef_i[:,j] for coef_i in coef] for j in range(nrep)]
    gramM = [tf.cast(tf.eye(nb_i), tf.float64) for nb_i in nbterms] # Gram matrix of the basis
    return(F, coef, gramM, basis)

def fproj_list_new(F_new, f_list):
    F, coef, gramM, basis = f_list
    nf  = len(F_new) # nb of functional inputs
    if (nf != len(F)):
         warnings.warn("The number of functional inputs len(F_new) should be equal to len(F).")

    # passing parameters of the projection to be used  
    F_full = [np.append(F[i], F_new[i], axis=1)  for i in range(nf)]    
    nrep_full = F_full[0].shape[1] # nb of new functional replicates
    coef_full = [tf.linalg.matmul(tf.transpose(basis[i]), F_full[i]) for i in range(nf)] # PCA coefficients
    coef_full = [[coef_i[:,j] for coef_i in coef_full] for j in range(nrep_full)]
    #print(coef_new)
    #coef_full = tf.stack([coef, coef_new])
    return(F_full, coef_full, gramM, basis)

  
def scaled_squared_euclid_fdist(f_list, f_list2, lengthscale, distances):
    """
    Returns ||(F_X - F'_X2ᵀ) / ℓ||² i.e. squared L2-norm.
    """
    def scaled_dist2ff(distance_f, lengthscale):
        # compute the scaled distance between two functional replicates
        scaled_distance = tf.reduce_sum(distance_f/lengthscale**2)
        return(tf.maximum(tf.sqrt(scaled_distance), 1e-40))
    
    if f_list2 is None:
        f_list2 = f_list
    #X, X2 = tf.cast(X, tf.int32), tf.cast(X2, tf.int32)

    coef, coef2 = f_list[1], f_list2[1] # coefficients of the projection
    gramM = f_list[2] # Gram matrix of the basis of the projection
    
    nrep, nrep2 = len(coef), len(coef2)
    if distances is None:
        dists2uu = [dist2ff(coef_i, coef_j, gramM) for coef_i in coef for coef_j in coef2]
    else:
        dists2uu = distances
    scaled_dists2uu = [scaled_dist2ff(dists2uu_i, lengthscale) for dists2uu_i in dists2uu]
    return(tf.reshape(scaled_dists2uu, [nrep, nrep2]))

def dist2ff(coeff, coeff2, gramM):
    # compute the scaled distance between two functional replicates
    diff_coef = [tf.reshape(x-y, [-1,1]) for x, y in zip(coeff, coeff2)]
    diff2ff = [tf.linalg.matmul(tf.linalg.matmul(x,y,True), x) for x, y in zip(diff_coef, gramM)]   
    return(tf.reshape(diff2ff, [-1]))







            