# scratch_pad
This repository contains little explorations in deep learning.

## Custom gradients
In custom_gradient_* modules I explore different ways of implementing custom gradients in Tensorflow, without having to create a Tensorflow op in C++ first.
You would want to implement your own gradient as opposed to relying on Tensorflow's automatic differentiation for reasons such as numerical stability.

### custom_gradient_with_py_func
In this approach we define a tf op using tf.py_func and assign a custom gradient function to it. tf.py_func is a wrapper for functions that have numpy inputs and runs on CPU.
### custom_gradient_with_python:
In this approach we use a workaround to define a custom gradient for a composition of Tensorflow ops.
### custom_gradient_with_eager:
This approach uses tensorflow.contrib.eager available as of Tensorflow 1.5 to define custom gradients for a composition of tensorflow ops.
