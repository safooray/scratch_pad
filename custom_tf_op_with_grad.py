import tensorflow as tf
import numpy as np


def python_func(x_in):
    return 2 * np.exp(x_in) - x_in


def grad_func(op, grad):
    # Return gradient wrt each input of the op
    return grad * ((2 * tf.exp(op.inputs[0])) - 1)


def my_py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def custom_tf_op(x, name=None):

    with tf.name_scope(name) as name:
        return my_py_func(python_func,
                          [x],
                          [tf.float32],
                          name=name,
                          grad=grad_func)[0]


if __name__=='__main__':

    with tf.Graph().as_default():
        x = tf.constant([[2.0, 4.0, 6.0], [6.0, 8.0, 10.0]], tf.float32, (2, 3),
                        'input')
        expected_op = tf.subtract(2 * tf.exp(x), x)
        expected_gr = tf.gradients(expected_op, [x])

        actual_op = custom_tf_op(x, name='my_op')
        actual_gr = tf.gradients(actual_op, [x])

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            actual_gr_val, expected_gr_val = sess.run([actual_gr, expected_gr])
            np.testing.assert_array_equal(expected_gr_val, actual_gr_val)
