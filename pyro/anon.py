from __future__ import absolute_import, division, print_function

import functools
import numbers

import numpy as np
import torch

import pyro

# Optional safety devices.
_CHECKING = []
_PYRO_PENDING = set()  # Set of sample and param objects.
_PYRO_BOUND = set()  # Set of addresses.

# This attempts to disallow a _Latent from storing an unregistered container
# that might accidentally hold unbound anon.sample or anon.param objects. If
# this ends up being too restrictive, we might drop this check.
_ALLOWED_TYPES = (
    type(None),
    bool,
    str,
    numbers.Number,
    torch.Tensor,
    torch.autograd.Variable,
    np.ndarray,
)


def function(fn):
    """
    Decorator for top-level model and guide functions.

    This adds an initial argument ``latent`` and adds error checking.
    """

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        if _CHECKING:
            raise RuntimeError('@functions do not support recursion')
        _CHECKING.append(None)
        _PYRO_PENDING.clear()
        _PYRO_BOUND.clear()
        latent = Latent('latent')
        try:
            result = fn(latent, *args, **kwargs)
        finally:
            _CHECKING.pop()
        if _PYRO_PENDING:
            raise RuntimeError('\n'.join(['Unbound sites:'] + list(map(str, _PYRO_PENDING))))
        return result

    return decorated


# Deferred sample site, will not run until stored in a _Latent.
class sample(object):
    __slots__ = ['fn', 'args', 'kwargs', 'address']
    __doc__ = pyro.sample.__doc__

    def __init__(self, fn, *args, **kwargs):
        if _CHECKING:
            _PYRO_PENDING.add(self)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.address = None

    def bind(self, address):
        if _CHECKING:
            _PYRO_PENDING.remove(self)
            _PYRO_BOUND.add(address)
        self.address = address
        return pyro.sample(address, self.fn, *self.args, **self.kwargs)


# Deferred observe site, will not run until stored in a _Latent.
@functools.wraps(pyro.observe)
def observe(fn, obs, *args, **kwargs):
    kwargs["obs"] = obs
    return sample(fn, *args, **kwargs)


# Deferred param site, will not run until stored in a _Latent.
class param(object):
    __slots__ = ['args', 'kwargs', 'address']
    __doc__ = pyro.sample.__doc__

    def __init__(self, *args, **kwargs):
        if _CHECKING:
            _PYRO_PENDING.add(self)
        self.args = args
        self.kwargs = kwargs
        self.address = None

    def bind(self, address):
        if _CHECKING:
            _PYRO_PENDING.remove(self)
            _PYRO_BOUND.add(address)
        self.address = address
        return pyro.param(address, *self.args, **self.kwargs)


class _Latent(object):
    """
    Base class for latent state containers.
    """
    def __init__(self, address='latent', replace=None):
        super(_Latent, self).__setattr__('_address', address)
        super(_Latent, self).__setattr__('_replace', replace)


class Latent(_Latent):
    """
    Object to hold latent state.
    """

    def __setattr__(self, name, value):
        address = '{}.{}'.format(self._address, name)
        if _CHECKING:
            if address in _PYRO_BOUND:
                raise RuntimeError('Cannot overwrite {}'.format(address))
        if isinstance(value, (sample, param)):
            value = value.bind(address)
        elif type(value) is object:
            value = Latent(address, lambda value: self.__setattr__(name, value))
        elif type(value) is dict:
            value = LatentDict(address, value)
        elif type(value) is list:
            value = LatentList(address, value)
        elif not isinstance(value, _ALLOWED_TYPES):
            raise TypeError('Latent cannot store objects of type {}'.format(type(value)))
        super(Latent, self).__setattr__(name, value)

    def __getattribute__(self, name):
        try:
            return super(Latent, self).__getattribute__(name)
        except AttributeError:
            address = '{}.{}'.format(self._address, name)
            value = Latent(address, lambda value: self.__setattr__(name, value))
            super(Latent, self).__setattr__(name, value)
            return value

    @functools.wraps(pyro.sample)
    def sample(self, fn, *args, **kwargs):
        value = pyro.sample(self._address, fn, *args, **kwargs)
        self._replace(value)
        return value

    @functools.wraps(pyro.observe)
    def observe(self, fn, obs, *args, **kwargs):
        value = pyro.observe(self._address, fn, obs, *args, **kwargs)
        self._replace(value)
        return value

    @functools.wraps(pyro.param)
    def param(self, *args, **kwargs):
        value = pyro.param(self._address, *args, **kwargs)
        self._replace(value)
        return value

    # TODO Make mutation methods safe.


class LatentDict(_Latent, dict):
    """
    Dict-like object to hold latent state.
    """

    def __init__(self, address, items):
        super(LatentDict, self).__init__(address)
        assert type(items) is dict
        for key, value in items:
            self[key] = value

    def __setitem__(self, key, value):
        address = '{}[{:r}]'.format(self._address, key)
        if _CHECKING:
            if address in _PYRO_BOUND:
                raise RuntimeError('Cannot overwrite {}'.format(address))
        if isinstance(value, (sample, param)):
            value = value.bind(address)
        elif type(value) is object:
            value = Latent(address, lambda value: self.__setitem__(key, value))
        elif type(value) is dict:
            value = LatentDict(address, value)
        elif type(value) is list:
            value = LatentList(address, value)
        elif not isinstance(value, _ALLOWED_TYPES):
            raise TypeError('LatentDict cannot store objects of type {}'.format(type(value)))
        super(LatentDict, self).__setitem__(key, value)

    # TODO Make mutation methods safe.


class LatentList(_Latent, list):
    """
    List-like object to hold latent state.
    """

    def __init__(self, address, values):
        super(LatentList, self).__init__(address)
        assert type(values) is list
        for value in values:
            self.append(value)

    def __setitem__(self, pos, value):
        address = '{}[{}]'.format(self._address, pos)
        if _CHECKING:
            if address in _PYRO_BOUND:
                raise RuntimeError('Cannot overwrite {}'.format(address))
        if isinstance(value, (sample, param)):
            value = value.bind(address)
        elif type(value) is object:
            value = Latent(address, lambda value: self.__setitem__(pos, value))
        elif type(value) is dict:
            value = LatentDict(address, value)
        elif type(value) is list:
            value = LatentList(address, value)
        elif not isinstance(value, _ALLOWED_TYPES):
            raise TypeError('LatentList cannot store objects of type {}'.format(type(value)))
        super(LatentList, self).__setitem__(pos, value)


    def add(self):
        pos = len(self)
        address = '{}.[{}]'.format(self._address, pos)
        value = Latent(address, lambda value: self.__setitem__(pos, value))
        super(LatentList, self).append(value)
        return value

    def append(self, value):
        pos = len(self)
        super(LatentList, self).append(None)
        self[pos] = value

    # TODO Make mutation methods safe.


def condition(fn, data):
    lifted_data = {}
    for k, v in data.items():
        lifted_data["latent." + k] = v
    return lambda latent, *args, **kwargs: pyro.condition(lift(fn, latent), data)(*args, **kwargs)


def lift(fn, _latent=None):
    given = _latent is not None
    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        # if _CHECKING:
        #     raise RuntimeError('@functions do not support recursion')
        _CHECKING.append(None)
        _PYRO_PENDING.clear()
        _PYRO_BOUND.clear()
        if not given:
            latent = Latent('latent')
        else:
            latent = _latent
        try:
            result = fn(latent, *args, **kwargs)
        finally:
            _CHECKING.pop()
        if _PYRO_PENDING:
            raise RuntimeError('\n'.join(['Unbound sites:'] + list(map(str, _PYRO_PENDING))))
        return result

    return decorated


def SVI(model, guide, *args, **kwargs):
    return pyro.infer.SVI(lift(model), lift(guide), *args, **kwargs)


def Importance(model, guide=None, *args, **kwargs):
    return pyro.infer.Importance(lift(model),
                                 lift(guide) if guide is not None else None,
                                 *args, **kwargs)

def Search(model, *args, **kwargs):
    return pyro.infer.Search(lift(model), *args, **kwargs)

def Marginal(*args, **kwargs):
    ret = pyro.infer.Marginal(*args, **kwargs)
    def guess(*args, **kwargs):
        d = ret(*args, **kwargs)
        l = Latent("latent")
        for k, v in d.items():
            l.__setattr__(k[len("latent."):], v)
        return l
    return guess
