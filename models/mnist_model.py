from base.base_model import BaseModel
import tensorflow as tf
from utils.config import data_type


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(data_type(),
            shape=(None, self.config.image_size, self.config.image_size, self.config.num_channels),
            name='image_input')
        self.y = tf.placeholder(tf.int64, shape=(None,))

        conv1_weights = tf.Variable(tf.truncated_normal([5, 5, self.config.num_channels, 32],
                                    stddev=0.1, seed=self.config.seed, dtype=data_type()))
        conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
        
        conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1,
                                                        seed=self.config.seed, dtype=data_type()))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
        
        fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([self.config.image_size // 4 * self.config.image_size // 4 * 64, 512],
                                stddev=0.1, seed=self.config.seed, dtype=data_type()))
        fc1_biases  = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
        
        fc2_weights = tf.Variable(tf.truncated_normal([512, self.config.num_labels],
                                                    stddev=0.1, seed=self.config.seed, dtype=data_type()))
        fc2_biases  = tf.Variable(tf.constant(0.1, shape=[self.config.num_labels], dtype=data_type()))

        conv1 = tf.nn.conv2d(self.x, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [tf.shape(pool2)[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        hidden = tf.nn.dropout(hidden, self.config.keep_prob, seed=1) # set dropout keep_prob
        logits = tf.matmul(hidden, fc2_weights) + fc2_biases
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases)
                      + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
       
        loss += 5e-4 * regularizers # Add the regularization term to the loss.

        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(0.01,  self.global_step_tensor, 1.0, 0.95, staircase=True)
        
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss, global_step=self.global_step_tensor)
        
        self.cross_entropy = loss
        self.train_step = optimizer

        correct_prediction = tf.nn.softmax(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

