from abc import ABC, abstractmethod
import scipy.stats as ss


class Distribution(ABC):

    @property
    @abstractmethod
    def distribution(self) -> 'scipy distribution':
        pass

    def p(self, x: float) -> float:
        """
        get P(X) for X = x
        :param x: value that takes x
        :return: P(X=x)
        """
        return self.distribution.pdf(x)

    def sample(self, size: int) -> 'np array':
        """
        get a sample of the distribution of size 'size'
        :param size: the size of the sample
        :return: array with the observations that are sample from the distribution
        """
        return self.distribution.rvs(size=size)


class UniformDistribution(Distribution):
    def __init__(self, min_: float = 0, max_: float = 1):
        """
        Construct an uniform distribution - [min,max]
        :param min_: minimum value of the distribution
        :param max_: maximum value of the distribution
        """
        self.min = min_
        self.max = max_

    @property
    def distribution(self):
        params = {"loc": self.min, "scale": self.max - self.min}
        return ss.uniform(**params)


class GaussianDistribution(Distribution):
    def __init__(self, mean: float = 0, std: float = 1):
        """
        Construct a normal distribution-N(mean, std^2)
        :param mean: Mean of the distribution
        :param std: standard deviation of the distribution
        """
        self.mean = mean
        self.std = std

    @property
    def distribution(self) -> 'scipy distribution':
        params = {"loc": self.mean, "scale": self.std}
        return ss.norm(**params)


class LaplaceDistribution(Distribution):
    def __init__(self, mean: float = 0, std: float = 1):
        """
        Construct a laplace distribution-N(mean, std^2)
        :param mean: Mean of the distribution
        :param std: standard deviation of the distribution
        """
        self.mean = mean
        self.std = std

    @property
    def distribution(self) -> 'scipy distribution':
        params = {"loc": self.mean, "scale": self.std}
        return ss.laplace(**params)


class BetaDistribution(Distribution):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """
        Construct a Beta distribution with alpha/beta values
        :param alpha: shape parameter
        :param beta: shape parameter
        """
        self.alpha = alpha
        self.beta = beta

    @property
    def distribution(self) -> 'scipy distribution':
        params = {"a": self.alpha, "b": self.beta}
        return ss.beta(**params)
