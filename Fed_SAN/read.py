import numpy as np
labels = list(np.load('label.npy'))
print(labels.count(1))
print(labels.count(-1))