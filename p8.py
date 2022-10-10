import numpy as np 

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class_targets = [0,1,1]

print(softmax_outputs[[0,1,2], class_targets])

"""np.clip() ===>>> Clip (limit) the values in an array.

Given an interval, values outside the interval
are clipped to the interval edges. For example, 
if an interval of [0, 1] is specified, values smaller
than 0 become 0, and values larger than 1 become 1."""

"""np.argmax() ===>>> Returns the indices of the maximum values along an axis."""