# -*- coding: utf-8 -*-
class _const:
    class ConstError(TypeError): pass
    class ConstCaseError(ConstError): pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value


NAME = _const()
NAME.PAD = "PAD"
NAME.START = "START"
NAME.END = "END"
NAME.UNK = "UNK"

TOKEN = _const()
TOKEN.PAD = 0 # Don't change!!!
TOKEN.START = 1
TOKEN.END = 2
TOKEN.UNK = 3
