from typing import Union

import tensorflow as tf
import gpflow
import numpy as np

from gpflow.utilities import positive
from gpflow.kernels import IndependentLatent, Combination, SeparateIndependent
from gpflow.kernels import LinearCoregionalization
from fkernels import *
from futil import *

from gpflow.conditionals.dispatch import conditional, sample_conditional
from gpflow.conditionals.mo_conditionals import coregionalization_conditional
from gpflow.inducing_variables import (SharedIndependentInducingVariables,
                                       SeparateIndependentInducingVariables,
                                       FallbackSeparateIndependentInducingVariables,
                                       FallbackSharedIndependentInducingVariables)

from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.conditionals.util import sample_mvn, mix_latent_gp

class fLinearCoregionalization(IndependentLatent, Combination):
    """
    Linear mixing of the latent GPs to form the output.
    """
    def __init__(self, kernels, lengthscale_f, f_list = None, f_list2 = None,
                distances = None, W = None, name=None):
        Combination.__init__(self, kernels, name)
        self.lengthscale_f = gpflow.Parameter(lengthscale_f, transform=positive())
        self.f_list = f_list # list with functional information
        self.f_list2 = f_list2 # list with functional information
        self.distances = distances # matrix with precomputed distances
        self.W = W

    def Kgg(self, X, X2):
        return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [L, N, N2]

    def K(self, X, X2=None, full_output_cov=True, presliced=False):
        Kxx = self.Kgg(X, X2) # [L, N, N2]
        
        r2 = scaled_squared_euclid_fdist(self.f_list, self.f_list2, self.lengthscale_f, self.distances)
        r = tf.sqrt(r2)
        sqrt5 = np.sqrt(5.)
        kf = (1. + sqrt5*r + 5.0/3.0*r2) * tf.exp(-sqrt5*r)
        self.W = tf.linalg.cholesky(kf)
        
        KxxW = Kxx[None, :, :, :] * self.W[:, :, None, None] # [P, L, N, N2]
        if full_output_cov:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
            WKxxW = tf.tensordot(self.W, KxxW, [[1], [1]])  # [P, P, N, N2]
            return tf.transpose(WKxxW, [2, 0, 3, 1])  # [N, P, N2, P]
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
            return tf.reduce_sum(self.W[:, :, None, None] * KxxW, [1])  # [P, N, N2]

    def K_diag(self, X, full_output_cov=True, presliced=False):
        K = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, L]
        
        r2 = scaled_squared_euclid_fdist(self.f_list, self.f_list2, self.lengthscale_f, self.distances)
        r = tf.sqrt(r2)
        sqrt5 = np.sqrt(5.)
        kf = (1. + sqrt5*r + 5.0/3.0*r2) * tf.exp(-sqrt5*r)

        self.W = tf.linalg.cholesky(kf)
        
        if full_output_cov:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # [N, P, P]
            Wt = tf.transpose(self.W) # [L, P]
            return tf.reduce_sum(K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :],
                                 axis=1)  # [N, P, P]
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # [N, P]
            return tf.linalg.matmul(K, self.W**2.0, transpose_b=True)  # [N, L]  *  [L, P]  ->  [N, P]


#class fLinearCoregionalization(IndependentLatent, Combination):
#    """
#    Linear mixing of the latent GPs to form the output.
#    """
#    def __init__(self, kernels, lengthscale_f, W = None, f_list = None, f_list2 = None,
#                 distances = None, name=None):
#        Combination.__init__(self, kernels, name)
##        fMatern52.__init__(self, variance=1.0, lengthscale=lengthscale_f,
##                           f_list = f_list, f_list2 = f_list2,
##                           distances = None, name = None)
#        self.lengthscale_f = gpflow.Parameter(lengthscale_f, transform=positive())
#        self.f_list = f_list # list with functional information
#        self.f_list2 = f_list2 # list with functional information
#        self.distances = distances # matrix with precomputed distances
#        self.W = W
#
#    def Kgg(self, X, X2):
#        return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [L, N, N2]
#
#    def K(self, X, X2=None, full_output_cov=True, presliced=False):
#        Kxx = self.Kgg(X, X2)  # [P, N, N2]
#        
##        r2 = scaled_squared_euclid_fdist(self.f_list, self.f_list2, self.lengthscale_f, self.distances)
##        r = tf.sqrt(r2)
##        sqrt5 = np.sqrt(5.)
##        kf = (1. + sqrt5*r + 5.0/3.0*r2) * tf.exp(-sqrt5*r)
#        
#        #self.W = np.random.randn(5, 5)#tf.cholinalg.cholesky(kf)
#        KxxW = Kxx[None, :, :, :] * self.W[:, :, None, None]  * self.lengthscale_f # [P, L, N, N2]
#        if full_output_cov:
#            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
#            WKxxW = tf.tensordot(self.W, KxxW, [[1], [1]])  # [P, P, N, N2]
#            return tf.transpose(WKxxW, [2, 0, 3, 1])  # [N, P, N2, P]
#        else:
#            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
#            return tf.reduce_sum(self.W[:, :, None, None] * KxxW, [1])  # [P, N, N2]
#
#        
#
#    def K_diag(self, X, full_output_cov=True, presliced=False):
#        K = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, P]
#        
##        r2 = scaled_squared_euclid_fdist(self.f_list, self.f_list, self.lengthscale_f, self.distances)
##        r = tf.sqrt(r2)
##        sqrt5 = np.sqrt(5.)
##        kf = (1. + sqrt5*r + 5.0/3.0*r2) * tf.exp(-sqrt5*r)
##        self.W = kf # [P, P]
##        
##        if full_output_cov:
##            # Can currently not use einsum due to unknown shape from `tf.stack()`
##            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # [N, P, P]
##            Wt = tf.transpose(self.W)  # [L, P]
##            return tf.reduce_sum(K[:, :, None] * Wt[None, :, :],
##                                 axis=1)  # [N, P, P]
##        else:
##            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # [N, P]
##            return tf.linalg.matmul(K, self.W, transpose_b=True)  # [N, P]  *  [P, P]  ->  [N, P]
#        #self.W = np.random.randn(5, 5)#tf.eye(kf.shape[0])#tf.linalg.cholesky(kf)
#        if full_output_cov:
#            # Can currently not use einsum due to unknown shape from `tf.stack()`
#            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # [N, P, P]
#            Wt = tf.transpose(self.W)  * self.lengthscale_f # [L, P]
#            return tf.reduce_sum(K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :],
#                                 axis=1)  # [N, P, P]
#        else:
#            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # [N, P]
#            return tf.linalg.matmul(K, self.W**2.0 * self.lengthscale_f, transpose_b=True)  # [N, L]  *  [L, P]  ->  [N, P]

#class fLinearCoregionalization(IndependentLatent, Combination):
#    """
#    Linear mixing of the latent GPs to form the output.
#    """
#    def __init__(self, kernels, W, a, name=None):
#        Combination.__init__(self, kernels, name)
#        self.a = gpflow.Parameter(a, transform=positive())        
#        self.W = gpflow.Parameter(W)  # [P, L]
#
#    def Kgg(self, X, X2):
#        return tf.stack([k.K(X, X2) * self.a for k in self.kernels], axis=0)  # [L, N, N2]
#
#    def K(self, X, X2=None, full_output_cov=True, presliced=False):
#        Kxx = self.Kgg(X, X2) # [L, N, N2]
#        KxxW = Kxx[None, :, :, :] * self.W[:, :, None, None] # [P, L, N, N2]
#        if full_output_cov:
#            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
#            WKxxW = tf.tensordot(self.W, KxxW, [[1], [1]])  # [P, P, N, N2]
#            return tf.transpose(WKxxW, [2, 0, 3, 1])  # [N, P, N2, P]
#        else:
#            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
#            return tf.reduce_sum(self.W[:, :, None, None] * KxxW, [1])  # [P, N, N2]
#
#    def K_diag(self, X, full_output_cov=True, presliced=False):
#        K = tf.stack([k.K_diag(X) * self.a for k in self.kernels], axis=1)  # [N, L]
#        if full_output_cov:
#            # Can currently not use einsum due to unknown shape from `tf.stack()`
#            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # [N, P, P]
#            Wt = tf.transpose(self.W) # [L, P]
#            return tf.reduce_sum(K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :],
#                                 axis=1)  # [N, P, P]
#        else:
#            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # [N, P]
#            return tf.linalg.matmul(K, self.W**2.0, transpose_b=True)  # [N, L]  *  [L, P]  ->  [N, P]

@Kuu.register(FallbackSeparateIndependentInducingVariables, fLinearCoregionalization)
def _Kuu(inducing_variable: FallbackSeparateIndependentInducingVariables,
         kernel: Union[SeparateIndependent, fLinearCoregionalization],
         *,
         jitter=0.0):
    Kmms = [Kuu(f, k) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)]
    Kmm = tf.stack(Kmms, axis=0)  # [L, M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat

@Kuf.register((FallbackSeparateIndependentInducingVariables, FallbackSharedIndependentInducingVariables),
              fLinearCoregionalization,
              object)
def _Kuf(inducing_variable: Union[SeparateIndependentInducingVariables, SharedIndependentInducingVariables],
         kernel: fLinearCoregionalization, Xnew: tf.Tensor):
    kuf_impl = Kuf.dispatch(type(inducing_variable), SeparateIndependent, object)
    K = tf.transpose(kuf_impl(inducing_variable, kernel, Xnew), [1, 0, 2])  # [M, L, N]
    return kernel.a * K[:, :, :, None] * tf.transpose(kernel.W)[None, :, None, :]  # [M, L, N, P]


@Kuf.register(SharedIndependentInducingVariables, fLinearCoregionalization, object)
def _Kuf(inducing_variable: SharedIndependentInducingVariables, kernel: SeparateIndependent, Xnew: tf.Tensor):
    return tf.stack([Kuf(inducing_variable.inducing_variable_shared, k, Xnew) for k in kernel.kernels], axis=0)  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, fLinearCoregionalization, object)
def _Kuf(inducing_variable, kernel, Xnew):
    return tf.stack([Kuf(f, k, Xnew)
                     for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)], axis=0)  # [L, M, N]

@conditional.register(object, (SharedIndependentInducingVariables, SeparateIndependentInducingVariables),
                      fLinearCoregionalization, object)
def coregionalization_conditional(
        Xnew, inducing_variable, kernel, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """Most efficient routine to project L independent latent gps through a mixing matrix W.
    The mixing matrix is a member of the `fLinearCoregionalization` and has shape [P, L].
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [L, M, M]
    - Kuf: [L, M, N]
    - Kff: [L, N] or [L, N, N]
    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.
    """
    ind_conditional = conditional.dispatch(object, SeparateIndependentInducingVariables, SeparateIndependent, object)
    gmu, gvar = ind_conditional(Xnew,
                                inducing_variable,
                                kernel,
                                f,
                                full_cov=full_cov,
                                q_sqrt=q_sqrt,
                                full_output_cov=False,
                                white=white)  # [N, L], [L, N, N] or [N, L]
    r2 = scaled_squared_euclid_fdist(kernel.f_list, kernel.f_list2, kernel.lengthscale_f, kernel.distances)
    r = tf.sqrt(r2)
    sqrt5 = np.sqrt(5.)
    kf = (1. + sqrt5*r + 5.0/3.0*r2) * tf.exp(-sqrt5*r)
    kernel.W = tf.linalg.cholesky(kf)
        
    return mix_latent_fgp(kernel.W, gmu, gvar, full_cov, full_output_cov)

def mix_latent_fgp(W, g_mu, g_var, full_cov, full_output_cov):
    r"""Takes the mean and variance of an uncorrelated L-dimensional latent GP
    and returns the mean and the variance of the mixed GP, `f = W g`,
    where both f and g are GPs, with W having a shape [P, L]
    :param W: [P, L]
    :param g_mu: [..., N, L]
    :param g_var: [..., N, L] (full_cov = False) or [L, ..., N, N] (full_cov = True)
    :return: f_mu and f_var, shape depends on `full_cov` and `full_output_cov`
    """
    f_mu = tf.tensordot(g_mu, W, [[-1], [-1]])  # [..., N, P]

    if full_cov and full_output_cov:  # g_var is [L, ..., N, N]
        # this branch is practically never taken
        g_var = rollaxis_left(g_var, 1)  # [..., N, N, L]
        g_var = tf.expand_dims(g_var, axis=-2)  # [..., N, N, 1, L]
        g_var_W = g_var * W  # [..., N, P, L]
        f_var = tf.tensordot(g_var_W, W, [[-1], [-1]])  # [..., N, N, P, P]
        f_var = leading_transpose(f_var, [..., -4, -2, -3, -1])  # [..., N, P, N, P]

    elif full_cov and not full_output_cov:  # g_var is [L, ..., N, N]
        # this branch is practically never taken
        f_var = tf.tensordot(g_var, W**2, [[0], [-1]])  # [..., N, N, P]
        f_var = leading_transpose(f_var, [..., -1, -3, -2])  # [..., P, N, N]

    elif not full_cov and full_output_cov:  # g_var is [..., N, L]
        g_var = tf.expand_dims(g_var, axis=-2)  # [..., N, 1, L]
        g_var_W = g_var * W  # [..., N, P, L]
        f_var = tf.tensordot(g_var_W, W, [[-1], [-1]])  # [..., N, P, P]

    elif not full_cov and not full_output_cov:  # g_var is [..., N, L]
        W_squared = W**2  # [P, L]
        f_var = tf.tensordot(g_var, W_squared, [[-1], [-1]])  # [..., N, P]

    return f_mu, f_var

#@sample_conditional.register(object, SharedIndependentInducingVariables, fLinearCoregionalization, object)
#def _sample_conditional(Xnew,
#                        inducing_variable,
#                        kernel,
#                        f,
#                        *,
#                        full_cov=False,
#                        full_output_cov=False,
#                        q_sqrt=None,
#                        white=False,
#                        num_samples=None):
#    """
#    `sample_conditional` will return a sample from the conditinoal distribution.
#    In most cases this means calculating the conditional mean m and variance v and then
#    returning m + sqrt(v) * eps, with eps ~ N(0, 1).
#    However, for some combinations of Mok and Mof more efficient sampling routines exists.
#    The dispatcher will make sure that we use the most efficent one.
#    :return: [N, P] (full_output_cov = False) or [N, P, P] (full_output_cov = True)
#    """
#    if full_cov:
#        raise NotImplementedError("full_cov not yet implemented")
#    if full_output_cov:
#        raise NotImplementedError("full_output_cov not yet implemented")
#
#    ind_conditional = conditional.dispatch(object, SeparateIndependentInducingVariables,
#                                           SeparateIndependent, object)
#    g_mu, g_var = ind_conditional(Xnew,
#                                  inducing_variable,
#                                  kernel,
#                                  f,
#                                  white=white,
#                                  q_sqrt=q_sqrt)  # [..., N, L], [..., N, L]
#    g_sample = sample_mvn(g_mu, g_var, "diag",
#                          num_samples=num_samples)  # [..., (S), N, L]
#    f_mu, f_var = mix_latent_gp(kernel.W, g_mu, g_var, full_cov,
#                                full_output_cov)
#    f_sample = tf.tensordot(g_sample, kernel.W, [[-1], [-1]])  # [..., N, P]
#    return f_sample, f_mu, f_var

