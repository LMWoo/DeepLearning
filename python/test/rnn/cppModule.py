import cpp as cpp

class cppModule(object):
    def __init__(self):
        pass

    def parameters(self):
        out = {}
        for key in self.__dict__:
            if isinstance(self.__dict__[key], cpp.cppRnn) or  isinstance(self.__dict__[key], cpp.cppLinear):
                out[key] = self.__dict__[key].parameters()
        return out    