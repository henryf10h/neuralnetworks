import numpy as np
import nnfs 

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.38],
                 [8.9, -.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values,axis=1,keepdims=True)#axis == 0 = columns

print(norm_values)

"""print(exp_values)

norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))"""
