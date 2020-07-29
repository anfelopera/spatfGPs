import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import positive
from gpflow.kernels.stationaries import square_distance 

from futil import *
from fkernels import *
from kronecker_ops import *

class fsMatern52(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, lengthscale_x=1.0, lengthscale_f=1.0,
                 f_list = None, f_list2 = None, **kwargs):
        for kwarg in kwargs:
            if kwarg not in {'name', 'active_dims'}:
                raise TypeError('Unknown keyword argument:', kwarg)
                
        super().__init__(**kwargs)
        #super().__init__(active_dims=[0])
        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscale_x = gpflow.Parameter(lengthscale_x, transform=positive())        
        self.lengthscale_f = gpflow.Parameter(lengthscale_f, transform=positive())
        self.f_list = f_list # list with functional information
        self.f_list2 = f_list2 # list with functional information

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        if not presliced:
            X, X2 = self.slice(X, X2)
        
        # kernel for functional inputs
        u, u2 = tf.unique(X[:,0])[0], tf.unique(X2[:,0])[0]
        if self.f_list2 is None:
            r2 = scaled_squared_euclid_fdist(u, u2, self.f_list, self.f_list, self.lengthscale_f)
        else:
            r2 = scaled_squared_euclid_fdist(u, u2, self.f_list, self.f_list2, self.lengthscale_f)
        r = tf.sqrt(r2)
        sqrt5 = np.sqrt(5.)
        kf = (1. + sqrt5*r + 5.0/3.0*r2) * tf.exp(-sqrt5*r)
        
        # spatial kernel
        x = tf.meshgrid(tf.unique(X[:,2])[0], tf.unique(X[:,1])[0])
        x = tf.stack([tf.reshape(x[1], [-1]), tf.reshape(x[0], [-1])], axis = 1)
        x2 = tf.meshgrid(tf.unique(X2[:,2])[0], tf.unique(X2[:,1])[0])
        x2 = tf.stack([tf.reshape(x2[1], [-1]), tf.reshape(x2[0], [-1])], axis = 1)
        
        r2 = square_distance(x/self.lengthscale_x, x2/self.lengthscale_x)
        r = tf.sqrt(r2)
        ks = self.variance * (1. + sqrt5*r + 5.0/3.0*r2) * tf.exp(-sqrt5*r)
        return kron_two(kf, ks)
    
    def K_diag(self, X, presliced=None):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))