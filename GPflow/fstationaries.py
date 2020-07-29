import numpy as np
import tensorflow as tf

#from ..base import Parameter
#from ..utilities import positive
#from .base import Kernel

from gpflow.base import Parameter
from gpflow.utilities import positive
from gpflow.kernels import Kernel

class fStationary(Kernel):
    """
    Base class for functional kernels that are stationary, that is, they only depend on

        r = || F - F' ||

    This class handles 'ard' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    functional input, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, variance=1.0, lengthscale=1.0, f_list=None):
        """
        :param variance: the (initial) value for the variance parameter
        :param lengthscale: the (initial) value for the lengthscale parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param f_list: list with information of the functional inputs
        """
       
        self.variance = Parameter(variance, transform=positive())
        self.lengthscale = Parameter(lengthscale, transform=positive())
        self.f_list = f_list # list with functional information

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscale.shape.ndims > 0
    
    def scaled_squared_euclid_fdist(X, X2, f_list, lengthscales):
        """
        Returns ||(F_X - F'_X2ᵀ) / ℓ||² i.e. squared L2-norm.
        """ 
        def dist2ff(coeff1, coeff2, gramM, lengthscales):
            diff_coef = [x-y for x, y in zip(coeff1, coeff2)]
            diff2ff = ([np.matmul(np.matmul(x, y), x) for x, y in zip(diff_coef, gramM)])
            return sum(diff2ff/lengthscales**2)
    
        nf = len(f_list[0])
        coef = f_list[1]
        gramM = f_list[2]
        dists2ff = [dist2ff(coef[i], coef[j], gramM, lengthscales) for i in X for j in X2]
        return np.asarray(dists2ff).reshape(nf, nf)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r2 = scaled_squared_euclid_fdist(X, X2, self.f_list, self.lengthscale)
        return self.K_r2(r2)

    def K_diag(self, X, presliced=False):
        return tf.fill(tf.shape(X), tf.squeeze(self.variance)) 

    def K_r2(self, r2):
        """
        Returns the kernel evaluated on r² (`r2`), which is the squared scaled Euclidean distance
        Should operate element-wise on r²
        """
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-40))
            return self.K_r(r)  # pylint: disable=no-member

        raise NotImplementedError


class SquaredExponential(fStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(r) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    def K_r2(self, r2):
        return self.variance * tf.exp(-0.5 * r2)


class RationalQuadratic(fStationary):
    """
    Rational Quadratic kernel,

    k(r) = σ² (1 + r² / 2αℓ²)^(-α)

    σ² : variance
    ℓ  : lengthscale
    α  : alpha, determines relative weighting of small-scale and large-scale fluctuations

    For α → ∞, the RQ kernel becomes equivalent to the squared exponential.
    """

    def __init__(self, variance=1.0, lengthscale=1.0, alpha=1.0, active_dims=None):
        super().__init__(variance=variance, lengthscale=lengthscale, active_dims=active_dims)
        self.alpha = Parameter(alpha, transform=positive())

    def K_r2(self, r2):
        return self.variance * (1 + 0.5 * r2 / self.alpha) ** (-self.alpha)


class Exponential(fStationary):
    """
    The Exponential kernel. It is equivalent to a Matern12 kernel with doubled lengthscales.
    """

    def K_r(self, r):
        return self.variance * tf.exp(-0.5 * r)


class Matern12(fStationary):
    """
    The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ² is the variance parameter
    """

    def K_r(self, r):
        return self.variance * tf.exp(-r)


class Matern32(fStationary):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
    """

    def K_r(self, r):
        sqrt3 = np.sqrt(3.)
        return self.variance * (1. + sqrt3 * r) * tf.exp(-sqrt3 * r)


class Matern52(fStationary):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is

    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
    """

    def K_r(self, r):
        sqrt5 = np.sqrt(5.)
        return self.variance * (1.0 + sqrt5 * r + 5.0 / 3.0 * tf.square(r)) * tf.exp(-sqrt5 * r)

