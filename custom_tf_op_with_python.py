import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


PROOF = 1
OP_NAME='my_op'


def python_func(x_in, name=None):
    with ops.name_scope(name):
        backward_func = tf.identity(x_in)
        forward_func = tf.subtract(2 * tf.exp(x_in), x_in)
        return backward_func + tf.stop_gradient(forward_func - backward_func) 


def grad_func(op, grad):
    # Return custom gradient wrt each input of the op.
    return grad * ((2 * tf.exp(op.inputs[0])) - 1) + PROOF


def my_op(func, inp, grad, name=None, victim_op='Identity'):
    # Need to generate a unique name to avoid duplicates.
    rnd_name = 'my_gradient' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({victim_op: rnd_name}):
        return func(inp, name=name)


if __name__=='__main__':
    with tf.Graph().as_default():
        x = tf.constant([[2.0, 4.0, 6.0], [6.0, 8.0, 10.0]], tf.float32, (2, 3),
                        'input')
        expected_op = tf.subtract(2 * tf.exp(x), x)
        expected_gr = tf.gradients(expected_op, [x])

        actual_op = my_op(python_func, x, grad_func, name=OP_NAME)
        actual_gr = tf.gradients(actual_op, [x])

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            actual_gr_val, expected_gr_val = sess.run([actual_gr, expected_gr])
            np.testing.assert_array_equal(expected_gr_val,
                                          [a - PROOF for a in actual_gr_val])
