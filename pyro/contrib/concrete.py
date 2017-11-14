from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma
from pyro.distributions.random_primitive import RandomPrimitive

class Concrete(Distribution):
    """Concrete distribution.

    Conditinuous version of a Categorical distribution using softmax
    relaxation of Gumbel-Max distribution. Returns a point in the
    simplex.

    Implementation based on [1]

    :param lambda_: Temperature parameter.
    :type lambda_: torch.autograd.Variable or scalar.
    :param alpha: A vector of location parameters. These should be non-negative.
    :type alpha: torch.autograd.Variable
    :param batch_size: Optional number of elements in the batch used to
        generate a sample. The batch dimension will be the leftmost dimension
        in the generated sample.
    :type batch_size: int

    [1] THE CONCRETE DISTRIBUTION: A CONTINUOUS RELAXATION OF DISCRETE RANDOM VARIABLES
    (Maddison et al, 2017)
    """
    reparameterized = True

    def __init__(self, temperature, ps=None, logits=None, batch_size=None, log_pdf_mask=None, *args,
                 **kwargs):
        self.temperature = temperature
        if (ps is None) == (logits is None):
            raise ValueError("Got ps={}, logits={}. Either `ps` or `logits` must be specified, "
                             "but not both.".format(ps, logits))
        self.ps, self.logits = get_probs_and_logits(ps=ps, logits=logits, is_multidimensional=True)
        self.log_pdf_mask = log_pdf_mask
        if self.ps.dim() == 1 and batch_size is not None:
            self.ps = self.ps.expand(batch_size, self.ps.size(0))
            self.logits = self.logits.expand(batch_size, self.logits.size(0))
            if log_pdf_mask is not None and log_pdf_mask.dim() == 1:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, log_pdf_mask.size(0))
        super(Concrete, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        ps = self.ps
        if x is not None:
            x = self._process_data(x)
            x_shape = x.shape if isinstance(x, np.ndarray) else x.size()
            try:
                ps = self.ps.expand(x_shape[:-event_dim] + self.event_shape())
            except RuntimeError as e:
                raise ValueError("Parameter `ps` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(ps.size(), x.size(), str(e)))
        return ps.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.ps.size()[-event_dim:]

    def shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.shape`
        """
        if self.one_hot:
            return self.batch_shape(x) + self.event_shape()
        return self.batch_shape(x) + (1,)

    def sample(self):
        """
        Draws either a single sample (if alpha.dim() == 1), or one sample per param (if alpha.dim() == 2).
        Reparameterized.
        """

        # Sample Gumbels, G_k = -log(-log(U))
        uniforms = torch.zeros(self.logits.data.size()).uniform_()
        eps = _get_clamping_buffer(uniforms)
        uniforms = uniforms.clamp(min=eps, max=1-eps)
        gumbels = Variable(uniforms.log().mul(-1).log().mul(-1))

        # Reparameterize
        z = F.logsoftmax((self.alpha.log() + gumbels) / self.temperature)
        return z if self.reparameterized else z.detach()

    def batch_log_pdf(self, x):
        """
        Evaluates log probability densities for one or a batch of samples and parameters.

        :return: tensor with log probabilities for each of the batches.
        :rtype: torch.autograd.Variable
        """
        n = self.event_shape()[0]
        logits = self.logits.expand(self.shape(x))
        log_scale = Variable(log_gamma(torch.Tensor([n]).expand(self.shape(x)))) - \
                             self.temperature.log().mul(-(n-1))
        scores = logits.log() + x.mul(-self.temperature)
        scores = scores.sum(dim=-1)

        log_part = n * logits.mul(x.mul(-self.temperature).exp()).sum(dim=-1).log()
        return (scores - log_part + log_scale).contiguous()

concrete = RandomPrimitive(Concrete)

class Exp(Bijector):
    def __init__(self):
        super(Exp, self).__init__()

    def __call__(self, x, *args, **kwargs):
        return torch.exp(x)

    def inverse(self, y, *args, **kwargs):
        eps = _get_clamping_buffer(y)
        return torch.log(y.clamp(min=eps))

    def log_det_jacobian(self, y, *args, **kwargs):
        return y.sum().abs()

class RelaxedCatergorical(Distribution):
    def __new__(cls, *args, **kwargs):
        return TransformedDistribution(RelaxedExpCategorical(*args, **kwargs), Exp())
