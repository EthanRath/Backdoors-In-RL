import torch
import numpy as np

class SingleValuePoison:
    def __init__(self, indices, value):
        self.indices = indices
        self.value = value

    def __call__(self, state):
        index = self.indices
        poisoned = torch.clone(state)
        if len(state.shape) > 1:
            poisoned[:, index] = self.value
        else:
            poisoned[index] = self.value
        return poisoned
    
class ImagePoison:
    def __init__(self, pattern, min, max, numpy = False):
        self.pattern = pattern
        self.min = min
        self.max = max
        self.numpy = numpy

    def __call__(self, state):
        if self.numpy:
            poisoned = np.float64(state)
            poisoned += self.pattern
            poisoned = np.clip(poisoned, self.min, self.max)
        else:
            poisoned = torch.clone(state)
            poisoned += self.pattern
            poisoned = torch.clamp(poisoned, self.min, self.max)
        return poisoned

class Discrete:
    def __init__(self, min = -1, max = 1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min = torch.tensor(min).to(device)
        self.max = torch.tensor(max).to(device)
        pass
    def __call__(self, target, action):
        return self.min if target != action else self.max
