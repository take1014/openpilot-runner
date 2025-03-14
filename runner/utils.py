import numpy as np

def safe_exp(x, out=None):
    return np.exp(np.clip(x, -np.inf, 11), out=out)

def sigmoid(x):
    return 1. / (1. + safe_exp(-x))

def softmax(x, axis=-1):
    x -= np.max(x, axis=axis, keepdim=True)
    if x.dtype == np.float32 or x.dtype == np.float64:
        safe_exp(x, out=x)
    else:
        x = safe_exp(x)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x