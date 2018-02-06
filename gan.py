import tensorflow as tf


slim = tf.contrib.slim

class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01


def generator(input, hidden_size):
    with tf.name_scope('generator'):
        h0 = slim.fully_connected(input, hidden_size, scope='g0'))
        h1 = slim.fully_connected(h0, 1, activation=None, scope='g1')
        return h1


def discriminator(input, hidden_size):
    with tf.name_scope('discriminator') as scope:
        h0 = slim.fully_connected(input, hidden_size), scope=scope)
        h1 = slim.fully_connected(h0, hidden_size, scope=scope)
        h2 = slim.fully_connected(h1, hidden_size, scope=scope)
        h3 = slim.fully_connected(h2, 1, activation=tf.sigmoid, scope=scope)
        return h3


def optimizer(loss, var_list):
    initial_learning_rate = 0.005
    decay = 0.95
    num_decay_steps = 150
    step = slim.get_or_create_global_step()

    lr = tf.train.exponential_decay(learning_rate=initial_learning_rate ,
                                    global_step=step,
                                    decay_steps=num_decay_steps,
                                    decay_rate=decay,
                                    staircase=True,
                                    name='learning_rate')

    train_op = tf.AdamOptimizer(learning_rate).minimize(loss,
                                                        global_step=step,
                                                        var_list=var_list)
    return train_op


class GAN(object):
    def __init__(self, params):
        with tf.variable_scope('G'):
            self.noise = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
            self.generator = generator(self.noise, params.hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.noise).
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))

        # Discriminator's decision on real data.
        with tf.variable_scope('D'):
            self.d_real = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch
            )

        # Discriminator's decision on fake/generated data.
        with tf.variable_scope('D', reuse=True):
            self.d_gen = discriminator(
                self.generator,
                params.hidden_size,
                params.minibatch
            )

        # Define the loss for discriminator and generator networks
        self.loss_d = tf.reduce_mean(-log(self.d_real) - log(1 - self.d_gen))
        self.loss_g = tf.reduce_mean(-log(self.d_gen))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


if __name__=='__main__':

