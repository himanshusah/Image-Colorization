import tensorflow as tf 

class cnn:
    def __init__():
        model()

    def input_tensor(input, labels):
        length = input.shape[1]
        breadth = input.shape[2]
        self.input = tf.placeholder(input, shape=(None, length, breadth), dtype=tf.float32)
        self.target = tf.placeholder(labels, shape=(None, length, breadth, 3), dtype=tf.float32)

    def model(is_training = False):        
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(self.input, 64, 3, padding='same', activation=tf.nn.leaky_relu)
        for layers in range(2, 16 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, 
                use_bias=False, activation=tf.nn.leaky_relu)

                output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    
        with tf.variable_scope('block17'):
            self.output = tf.layers.conv2d(output, 3, 3, padding='same')

    def find_loss(predicted_images, label_images):
        

    def train():


    def statistics():
        return acc, loss
            
    return output

