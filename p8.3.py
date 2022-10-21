import numpy as np
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

print(class_targets.shape)
print(len(class_targets.shape))
print(range(len(softmax_outputs)))
print(softmax_outputs[range(len(softmax_outputs)),class_targets]) # https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8

# Probabilities for target values -
# only if categorical labels

if len(class_targets.shape) == 1:
   correct_confidences = softmax_outputs[
   range(len(softmax_outputs)),
   class_targets]

# Mask values - only for one-hot encoded labels
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
    softmax_outputs * class_targets,
    axis=1)

# Losses
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)
print(average_loss)