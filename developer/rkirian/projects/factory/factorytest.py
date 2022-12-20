import abc
class Getter(abc.ABC):
    def __init__(self):
        pass
    def __new__(cls, *args, **kwargs):
        print('Getter.__new__')
        self = object.__new__(cls)
        self._args = args
        self._kwargs = kwargs
        return self
    def factory(self):
        cls = type(self)
        args = self._args
        kwargs = self._kwargs
        def f():
            return cls(*args, **kwargs)
        return f
class Getter2(Getter):
    def __init__(self, a, b=None):
        print('Getter2.__init__')
        super().__init__()
        self.a = a
        self.b = b
getter = Getter2('a', b='b')
factory = getter.factory()
f1 = factory()
f1.a = 'c'
f2 = factory()
print(type(f1), f1.__dict__)
print(type(f2), f2.__dict__)
