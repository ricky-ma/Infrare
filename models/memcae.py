import tensorflow as tf
tf.config.run_functions_eagerly(True)


class MemCAE(tf.keras.Model):
    """Memory-augmented convolutional autoencoder."""

    def __init__(self, latent_dim, training, input_shape, batch_size):
        super(MemCAE, self).__init__()
        self.architecture = "MemCAE"
        self.latent_dim = latent_dim
        self.training = training
        self.input_img_shape = input_shape
        self.batch_size = batch_size
        self.name_bank, self.params_trainable, self.conv_shapes = [], [], []
        self.initializer = tf.initializers.glorot_normal()

    # @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=self.input_img_shape)
        return self.decode(eps)

    def encode(self, x):
        self.conv_shapes.append(x.shape)

        conv1 = self.conv2d(inputs=x, stride=2, padding='SAME',
                            variables=self.get_weight(vshape=[1, 1, 3, 16], name="encode1"))
        bn1 = self.batch_normalization(x=conv1, name="bn1")
        act1 = tf.compat.v1.nn.relu(bn1)
        self.conv_shapes.append(act1.shape)

        conv2 = self.conv2d(inputs=act1, stride=2, padding='SAME',
                            variables=self.get_weight(vshape=[3, 3, 16, 32], name="encode2"))
        bn2 = self.batch_normalization(x=conv2, name="bn2")
        act2 = tf.compat.v1.nn.relu(bn2)
        self.conv_shapes.append(act2.shape)

        conv3 = self.conv2d(inputs=act2, stride=2, padding='SAME',
                            variables=self.get_weight(vshape=[3, 3, 32, 64], name="encode3"))
        bn3 = self.batch_normalization(x=conv3, name="bn3")
        z = tf.compat.v1.nn.relu(bn3)

        _, _, _, c = z.shape
        return z, c

    def memory(self, z, c):
        N = 2000
        w_memory = self.get_weight(vshape=[1, 1, N, c], bias=False, name="memory")
        cosim = self.cosine_sim(x1=z, x2=w_memory)  # Eq.5
        attention = tf.nn.softmax(cosim)  # Eq.4

        lam = 1 / N  # deactivate the 1/N of N memories.

        addr_num = tf.keras.activations.relu(attention - lam) * attention
        addr_denum = tf.abs(attention - lam) + 1e-12
        memory_addr = addr_num / addr_denum

        renorm = tf.clip_by_value(memory_addr, 1e-12, 1 - 1e-12)

        z_hat = tf.linalg.matmul(renorm, w_memory)
        return z_hat, renorm

    def decode(self, z_hat):
        _, h, w, c = self.conv_shapes[-1]
        convt1 = self.conv2d_tr(inputs=z_hat, stride=2, padding='SAME', output_shape=[self.batch_size, h, w, c],
                                variables=self.get_weight(vshape=[3, 3, c, 64], transpose=True, name="decode1"))
        bnt1 = self.batch_normalization(x=convt1, name="bnt1")
        actt1 = tf.compat.v1.nn.relu(bnt1)

        _, h, w, c = self.conv_shapes[-2]
        convt2 = self.conv2d_tr(inputs=actt1, stride=2, padding='SAME', output_shape=[self.batch_size, h, w, c],
                                variables=self.get_weight(vshape=[3, 3, c, 32], transpose=True, name="decode2"))
        bnt2 = self.batch_normalization(x=convt2, name="bnt2")
        actt2 = tf.compat.v1.nn.relu(bnt2)

        _, h, w, c = self.conv_shapes[-3]
        convt3 = self.conv2d_tr(inputs=actt2, stride=2, padding='SAME', output_shape=[self.batch_size, h, w, c],
                                variables=self.get_weight(vshape=[3, 3, c, 16], transpose=True, name="decode3"))

        x_hat = tf.compat.v1.clip_by_value(convt3, 1e-12, 1 - 1e-12)
        return x_hat

    def compute_loss(self, x):
        z, c = self.encode(x)
        z_hat, w_hat = self.memory(z=z, c=c)
        x_hat = self.decode(z_hat)
        mem_ent = tf.reduce_sum((-w_hat) * tf.math.log(w_hat + 1e-12), axis=(1, 2, 3))
        mse = tf.reduce_sum(tf.square(x - x_hat), axis=(1, 2, 3))
        return tf.reduce_mean(mse + (0.0002 * mem_ent))

    def cosine_sim(self, x1, x2):
        num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
        denom = tf.linalg.matmul(x1 ** 2, tf.transpose(x2, perm=[0, 1, 3, 2]) ** 2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)
        return w

    def conv2d(self, inputs, variables, stride, padding):
        [weights, biases] = variables
        out = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding)
        return out + biases

    def conv2d_tr(self, inputs, variables, output_shape, stride, padding):
        [weights, biases] = variables
        out = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1, stride, stride, 1], padding=padding)
        return out + biases

    def batch_normalization(self, x, name=""):
        bnlayer = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            renorm_momentum=0.99,
            trainable=True,
            name="%s_bn" % name,
        )
        bn = bnlayer(inputs=x, training=self.training)
        return bn

    def get_weight(self, vshape, transpose=False, bias=True, name=""):
        try:
            idx_w = self.name_bank.index("%s_w" % name)
            if bias:
                idx_b = self.name_bank.index("%s_b" % name)
        except:
            w = tf.Variable(self.initializer(vshape), name="%s_w" % name, trainable=True, dtype=tf.float32)
            self.name_bank.append("%s_w" % name)
            self.params_trainable.append(w)
            if bias:
                if transpose:
                    b = tf.Variable(self.initializer([vshape[-2]]),
                                    name="%s_b" % name, trainable=True, dtype=tf.float32)
                else:
                    b = tf.Variable(self.initializer([vshape[-1]]),
                                    name="%s_b" % name, trainable=True, dtype=tf.float32)
                self.name_bank.append("%s_b" % name)
                self.params_trainable.append(b)
        else:
            w = self.params_trainable[idx_w]
            if bias:
                b = self.params_trainable[idx_b]
        if bias:
            return w, b
        else:
            return w
