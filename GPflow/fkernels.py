import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import positive
from futil import *

class fSquaredExponential(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, lengthscale=1.0, f_list = None, f_list2 = None,
                 distances = None, **kwargs):
        for kwarg in kwargs:
            if kwarg not in {'name', 'active_dims'}:
                raise TypeError('Unknown keyword argument:', kwarg)
                
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive())
        self.f_list = f_list # list with functional information
        self.f_list2 = f_list2 # list with functional information
        self.distances = distances # matrix with precomputed distances

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        if not presliced:
            X, X2 = self.slice(X, X2)
        # converting inputs to integers
        X, X2 = tf.cast(X, tf.int32), tf.cast(X2, tf.int32)
        
        # computing distances between functional spaces
        r2 = scaled_squared_euclid_fdist(self.f_list, self.f_list2, self.lengthscale, self.distances)
        # computing the kernel between functional spaces
        kuu = self.variance * tf.exp(-0.5 * r2)
        
        # computing the expanded kernel according to functional indices
        idx2, idx = tf.meshgrid(X2, X)
        kff = tf.gather_nd(kuu, tf.stack((idx, idx2), -1))
        return kff       

    def K_diag(self, X, presliced=None):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    
class fExponential(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, lengthscale=1.0, f_list = None, f_list2 = None,
                 distances = None, **kwargs):
        for kwarg in kwargs:
            if kwarg not in {'name', 'active_dims'}:
                raise TypeError('Unknown keyword argument:', kwarg)
                
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive())
        self.f_list = f_list # list with functional information
        self.f_list2 = f_list2 # list with functional information
        self.distances = distances # matrix with precomputed distances

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        if not presliced:
            X, X2 = self.slice(X, X2)
        # converting inputs to integers
        X, X2 = tf.cast(X, tf.int32), tf.cast(X2, tf.int32)
        
        # computing distances between functional spaces
        r2 = scaled_squared_euclid_fdist(X, X2, self.f_list, self.f_list2, self.lengthscale, self.distances)
        r = tf.sqrt(r2)
        # computing the kernel between functional spaces
        kuu = self.variance * tf.exp(-0.5 * r)
       
        # computing the expanded kernel according to functional indices
        idx2, idx = tf.meshgrid(X2, X)
        kff = tf.gather_nd(kuu, tf.stack((idx, idx2), -1))
        return kff        

    def K_diag(self, X, presliced=None):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    
class fMatern32(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, lengthscale=1.0, f_list = None, f_list2 = None,
                 distances = None, **kwargs):
        for kwarg in kwargs:
            if kwarg not in {'name', 'active_dims'}:
                raise TypeError('Unknown keyword argument:', kwarg)
                
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive())
        self.f_list = f_list # list with functional information
        self.f_list2 = f_list2 # list with functional information
        self.distances = distances # matrix with precomputed distances

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        if not presliced:
            X, X2 = self.slice(X, X2)
       # converting inputs to integers
        #X, X2 = tf.cast(X, tf.int32), tf.cast(X2, tf.int32)
        
        # computing distances between functional spaces
        r2 = scaled_squared_euclid_fdist(self.f_list, self.f_list2, self.lengthscale, self.distances)
        r = tf.sqrt(r2)
        # computing the kernel between functional spaces
        sqrt3 = np.sqrt(3.)
        kuu = self.variance * (1. + sqrt3 * r) * tf.exp(-sqrt3 * r)
        
        # computing the expanded kernel according to functional indices
        idx2, idx = tf.meshgrid(X2, X)
        kff = tf.gather_nd(kuu, tf.stack((idx, idx2), -1))
        return kff

    def K_diag(self, X, presliced=None):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    
class fMatern52(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, lengthscale=1.0, f_list = None, f_list2 = None,
                 distances = None, **kwargs):
        for kwarg in kwargs:
            if kwarg not in {'name', 'active_dims'}:
                raise TypeError('Unknown keyword argument:', kwarg)
                
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive())
        self.f_list = f_list # list with functional information
        self.f_list2 = f_list2 # list with functional information
        self.distances = distances # matrix with precomputed distances

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        if not presliced:
            X, X2 = self.slice(X, X2)
        # converting inputs to integers
        X, X2 = tf.cast(X, tf.int32), tf.cast(X2, tf.int32)
        
        # computing distances between functional spaces
        r2 = scaled_squared_euclid_fdist(self.f_list, self.f_list2, self.lengthscale, self.distances)
        r = tf.sqrt(r2)
        # computing the kernel between functional spaces
        sqrt5 = np.sqrt(5.)
        kuu = self.variance * (1. + sqrt5*r + 5.0/3.0*r2) * tf.exp(-sqrt5*r)
        
        # computing the expanded kernel according to functional indices
        idx2, idx = tf.meshgrid(X2, X)
        kff = tf.gather_nd(kuu, tf.stack((idx, idx2), -1))
        return kff
    
    def K_diag(self, X, presliced=None):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))