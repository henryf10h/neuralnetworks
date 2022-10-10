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

#we want to normalize data , we need to exponentiate them too.
# why exp values? because we need probability distribution (-oo, 1)
# why softmax and not ReLU? because we want to pass negatives, RElU transforms negatives to zero
# why to normalize them? to get the actual distribution of probabilities