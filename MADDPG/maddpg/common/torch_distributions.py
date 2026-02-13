import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import maddpg.common.torch_util as U

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def logp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return U.Placeholder(prepend_shape + self.param_shape(), dtype=torch.float32, name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return U.Placeholder(prepend_shape + self.sample_shape(), dtype=self.sample_dtype(), name=name)

class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return torch.long

class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return SoftCategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return [self.ncat]
    def sample_dtype(self):
        return torch.float32

class MultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return torch.long

class SoftMultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return SoftMultiCategoricalPd
    def pdfromflat(self, flat):
        return SoftMultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [sum(self.ncats)]
    def sample_dtype(self):
        return torch.float32

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd
    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return torch.float32

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return BernoulliPd
    def param_shape(self):
        return [self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return torch.long

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return torch.argmax(self.logits, dim=1)
    def logp(self, x):
        return -F.cross_entropy(self.logits, x, reduction='none')
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, dim=1, keepdim=True)[0]
        a1 = other.logits - U.max(other.logits, dim=1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = U.sum(ea0, dim=1, keepdim=True)
        z1 = U.sum(ea1, dim=1, keepdim=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, dim=1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = U.sum(ea0, dim=1, keepdim=True)
        p0 = ea0 / z0
        return U.sum(p0 * (torch.log(z0) - a0), dim=1)
    def sample(self):
        u = torch.rand(self.logits.shape, device=self.logits.device)
        return U.argmax(self.logits - torch.log(-torch.log(u)), dim=1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftCategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return F.softmax(self.logits, dim=-1)
    def logp(self, x):
        return -F.cross_entropy(self.logits, x, reduction='none')
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, dim=1, keepdim=True)[0]
        a1 = other.logits - U.max(other.logits, dim=1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = U.sum(ea0, dim=1, keepdim=True)
        z1 = U.sum(ea1, dim=1, keepdim=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, dim=1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = U.sum(ea0, dim=1, keepdim=True)
        p0 = ea0 / z0
        return U.sum(p0 * (torch.log(z0) - a0), dim=1)
    def sample(self):
        u = torch.rand(self.logits.shape, device=self.logits.device)
        return F.softmax(self.logits - torch.log(-torch.log(u)), dim=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MultiCategoricalPd(Pd):
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = torch.tensor(low, dtype=torch.long, device=flat.device)
        self.categoricals = []
        split_sizes = high - low + 1
        split_idx = 0
        for size in split_sizes:
            logits = flat[:, split_idx:split_idx+size]
            self.categoricals.append(CategoricalPd(logits))
            split_idx += size
    def flatparam(self):
        return self.flat
    def mode(self):
        modes = [p.mode() for p in self.categoricals]
        result = self.low + torch.stack(modes, dim=-1).long()
        return result
    def logp(self, x):
        x_shifted = x - self.low
        logps = []
        for i, (p, px) in enumerate(zip(self.categoricals, torch.unbind(x_shifted, dim=1))):
            logps.append(p.logp(px))
        return torch.stack(logps, dim=0).sum(dim=0)
    def kl(self, other):
        kls = []
        for p, q in zip(self.categoricals, other.categoricals):
            kls.append(p.kl(q))
        return torch.stack(kls, dim=0).sum(dim=0)
    def entropy(self):
        entropies = [p.entropy() for p in self.categoricals]
        return torch.stack(entropies, dim=0).sum(dim=0)
    def sample(self):
        samples = [p.sample() for p in self.categoricals]
        result = self.low + torch.stack(samples, dim=-1).long()
        return result
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftMultiCategoricalPd(Pd):
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = torch.tensor(low, dtype=torch.float32, device=flat.device)
        self.categoricals = []
        split_sizes = high - low + 1
        split_idx = 0
        for size in split_sizes:
            logits = flat[:, split_idx:split_idx+size]
            self.categoricals.append(SoftCategoricalPd(logits))
            split_idx += size
    def flatparam(self):
        return self.flat
    def mode(self):
        x = []
        for i, p in enumerate(self.categoricals):
            x.append(self.low[i] + p.mode())
        return torch.cat(x, dim=-1)
    def logp(self, x):
        x_shifted = x - self.low
        logps = []
        for i, (p, px) in enumerate(zip(self.categoricals, torch.unbind(x_shifted, dim=1))):
            logps.append(p.logp(px))
        return torch.stack(logps, dim=0).sum(dim=0)
    def kl(self, other):
        kls = []
        for p, q in zip(self.categoricals, other.categoricals):
            kls.append(p.kl(q))
        return torch.stack(kls, dim=0).sum(dim=0)
    def entropy(self):
        entropies = [p.entropy() for p in self.categoricals]
        return torch.stack(entropies, dim=0).sum(dim=0)
    def sample(self):
        x = []
        for i, p in enumerate(self.categoricals):
            x.append(self.low[i] + p.sample())
        return torch.cat(x, dim=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = torch.chunk(flat, 2, dim=1)
        self.mean = mean
        self.logstd = logstd
        self.std = torch.exp(logstd)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean
    def logp(self, x):
        return - 0.5 * U.sum(torch.square((x - self.mean) / self.std), dim=1) \
               - 0.5 * np.log(2.0 * np.pi) * x.shape[1] \
               - U.sum(self.logstd, dim=1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return U.sum(other.logstd - self.logstd + (torch.square(self.std) + torch.square(self.mean - other.mean)) / (2.0 * torch.square(other.std)) - 0.5, dim=1)
    def entropy(self):
        return U.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), dim=1)
    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = torch.sigmoid(logits)
    def flatparam(self):
        return self.logits
    def mode(self):
        return torch.round(self.ps).long()
    def logp(self, x):
        return - U.sum(F.binary_cross_entropy_with_logits(self.logits, x.float(), reduction='none'), dim=1)
    def kl(self, other):
        return U.sum(F.binary_cross_entropy_with_logits(other.logits, self.ps, reduction='none'), dim=1) - U.sum(F.binary_cross_entropy_with_logits(self.logits, self.ps, reduction='none'), dim=1)
    def entropy(self):
        return U.sum(F.binary_cross_entropy_with_logits(self.logits, self.ps, reduction='none'), dim=1)
    def sample(self):
        p = torch.sigmoid(self.logits)
        u = torch.rand_like(p)
        return (u < p).long()
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(ac_space):
    try:
        from gym import spaces
    except ImportError:
        # Handle case where gym is not available
        raise NotImplementedError("Gym spaces not available")

    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        # return CategoricalPdType(ac_space.n)
        return SoftCategoricalPdType(ac_space.n)
    elif hasattr(ac_space, '__class__') and 'MultiDiscrete' in str(ac_space.__class__):
        #return MultiCategoricalPdType(ac_space.low, ac_space.high)
        return SoftMultiCategoricalPdType(ac_space.low, ac_space.high)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError(f"Unsupported action space type: {type(ac_space)}")

def shape_el(v, i):
    if len(v.shape) > i and v.shape[i] is not None:
        return v.shape[i]
    else:
        return v.size(i) if hasattr(v, 'size') else None