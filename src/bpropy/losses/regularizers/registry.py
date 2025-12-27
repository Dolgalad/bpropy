REGULARIZER_REGISTRY = {}

def register_regularizer(name):
    def deco(fn):
        REGULARIZER_REGISTRY[name] = fn
        return fn
    return deco


