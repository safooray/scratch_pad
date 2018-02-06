import tensorflow as tf
import tensorflow.contrib.eager as tfe

PROOF = 1

@tfe.custom_gradient
def python_func(x_in):
    def grad_func(grad):
        return grad * ((2 * tf.exp(x_in)) - 1) + PROOF

    forward_func = tf.subtract(2 * tf.exp(x_in), x_in)
    return forward_func, grad_func

if __name__=='__main__':
    with tf.Graph().as_default():
        x = tf.constant([[2.0, 4.0, 6.0], [6.0, 8.0, 10.0]], tf.float32, (2, 3),
                        'input')
        expected_op = tf.subtract(2 * tf.exp(x), x)
        expected_gr = tf.gradients(expected_op, [x])

        actual_op = python_func(x)
        actual_gr = tf.gradients(actual_op, [x])

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            actual_gr_val, expected_gr_val = sess.run([actual_gr, expected_gr])
            np.testing.assert_array_equal(expected_gr_val,
                                          [a - PROOF for a in actual_gr_val])
