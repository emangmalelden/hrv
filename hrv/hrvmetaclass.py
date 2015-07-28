import numpy as np

class SignalMeta(type):

    def __init__(cls, name, basis, dic):
        super(SignalMeta, cls).__init__(name, basis, dic)
        for key, attr in dic.items():
            if hasattr(attr, "set_name"):
                attr.set_name("__" + name, key)


class Signal(object):

    def __init__(self):
        self.set_name(self.__class__.__name__, id(self))

    def set_name(self, prefix, key):
        self.target_name = "%s_%s" % (prefix, key)

    def __get__(self, instance, owner):
        return getattr(instance, self.target_name)

    def __set__(self, instance, signal):
        if not isinstance(signal, list) and \
                not isinstance(signal, np.ndarray):
            raise ValueError("rri must be a list or a numpy array!")
        else:
            if not all(isinstance(val, int) or isinstance(val, float) or
                    isinstance(val, np.generic) for val in signal):
                raise ValueError("rri must be a list or a numpy array!")
            else:
                setattr(instance, self.target_name, np.array(signal))

class Positive(object):

    def __init__(self):
        self.set_name(self.__class__.__name__, id(self))

    def set_name(self, prefix, key):
        self.target_name = "%s_%s" % (prefix, key)

    def __get__(self, instance, owner):
        return getattr(instance, self.target_name)

    def __set__(self, instance, val):
        if not isinstance(val, int) and not isinstance(val, float) and \
                not isinstance(val, np.generic):
            raise ValueError("Value must be a positive number!")
        else:
            if val < 0:
                raise ValueError("Value must be a positive number!")
            else:
                setattr(instance, self.target_name, val)

class MetaModel(object):
    __metaclass__ = SignalMeta
