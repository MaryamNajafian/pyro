from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import softmax, log_gamma
from pyro.distributions.random_primitive import RandomPrimitive

class Concrete(Distribution):
    """Concrete distribution.

    Conditinuous version of a Categorical distribution using softmax
    relaxation of Gumbel-Max distribution. Returns a point in the
    simplex.

    Implementation based on [1]

    :param alpha: A vector of location parameters. These should be non-negative.
    :type alpha: torch.autograd.Variable
    :param lambda_: Temperature parameter.
    :type lambda_: torch.autograd.Variable or scalar.
    :param batch_size: Optional number of elements in the batch used to
        generate a sample. The batch dimension will be the leftmost dimension
        in the generated sample.
    :type batch_size: int

    [1] THE CONCRETE DISTRIBUTION: A CONTINUOUS RELAXATION OF DISCRETE RANDOM VARIABLES
    (Maddison et al, 2017)
    """
    reparameterized = True

    def __init__(self, alpha, lambd, batch_size=None, *args, **kwargs):
        self.alpha = alpha
        self.lambd = lambd
        if alpha.dim() not in (1, 2):
            raise ValueError("Parameter alpha must be either 1 or 2 dimensional.")
        if alpha.dim() == 1 and batch_size is not None:
            self.alpha = alpha.expand(batch_size, alpha.size(0))
        super(Concrete, self).__init__(*args, **kwargs)

    def _process_data(self, x):
        if x is not None:
            if isinstance(x, list):
                x = np.array(x)
            elif not isinstance(x, (Variable, torch.Tensor, np.ndarray)):
                raise TypeError(("Data should be of type: list, Variable, Tensor, or numpy array"
                                 "but was of {}".format(str(type(x)))))
        return x

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        alpha = self.alpha
        if x is not None:
            if x.size()[-event_dim] != alpha.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.alpha.size()[-1], but got {} vs {}".format(
                                     x.size(-1), alpha.size(-1)))
            try:
                alpha = self.alpha.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `alpha` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(alpha.size(), x.size(), str(e)))
        return alpha.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        return self.alpha.size()[-1:]


    def sample(self):
        """
        Draws either a single sample (if alpha.dim() == 1), or one sample per param (if alpha.dim() == 2).
        Reparameterized.
        """

        # Sample Gumbels, G_k = -log(-log(U))
        n = self.event_shape()
        gumbel = Variable(torch.rand(n).type_as(self.alpha.data)
                          .log().mul(-1).log().mul(-1))

        # Reparameterize
        z = softmax((self.alpha.log() + gumbel) / self.lambd)
        return z if self.reparameterized else z.detach()

    def batch_log_pdf(self, x):
        """
        Evaluates log probability densities for one or a batch of samples and parameters.

        :return: tensor with log probabilities for each of the batches.
        :rtype: torch.autograd.Variable
        """
        n = self.event_shape()[0]
        alpha = self.alpha.expand(self.shape(x))
        log_scale = Variable(log_gamma(torch.Tensor([n])) - np.log(self.lambd) * (-(n-1)))
        scores = x.log().mul(-self.lambd-1) + alpha.log()
        scores = scores.sum(dim=-1)
        log_part = n * alpha.mul(x.pow(-self.lambd)).sum(dim=-1).log()
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return (scores - log_part + log_scale).contiguous().view(batch_log_pdf_shape)


    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.alpha

concrete = RandomPrimitive(Concrete)
