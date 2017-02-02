import tensorflow as tf
import numpy as np
from svhn_data import get_test_data, train_and_validation_data,get_train_data

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 2) == np.argmax(labels, 2))/ predictions.shape[0])/6

offset = 0

image_height = 32
image_width = 128
num_channels = 1
num_labels = 11
num_digits = 6

batch_size = 128
patch_size = 5
depth = 256
num_hidden1 = 2048
num_hidden2 = 512

Xvalidation, yvalidation = get_train_data('train/',offset,batch_size)
offset += batch_size

#print dimensions
print("Xvalidation shape is {} and yvalidation shape is {}".format(Xvalidation.shape, yvalidation.shape))

#build a graph
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, num_channels))
    tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_digits, num_labels))
    tf_valid_dataset = tf.cast(tf.constant(Xvalidation),tf.float32)
    #tf_test_dataset = tf.constant(Xtest)

    #Create variables
    #convolutions layer 1
    layer1_W = weight_variable([patch_size, patch_size, num_channels, depth//2])
    layer1_bias = bias_variable([depth//2])

    #layer2
    layer2_W = weight_variable([patch_size, patch_size, depth//2, depth//2])
    layer2_bias = bias_variable([depth//2])

    #layer3
    layer3_W = weight_variable([patch_size, patch_size, depth//2, depth])
    layer3_bias = bias_variable([depth])

    #layer3a
    layer3a_W = weight_variable([patch_size, patch_size, depth, depth])
    layer3a_bias = bias_variable([depth])

    #layer3
    layer3c_W = weight_variable([patch_size, patch_size, depth, depth])
    layer3c_bias = bias_variable([depth])
    

    #later Insert Regression head here
    #layer 4
    layer4_W = weight_variable([image_height//8*image_width//8*depth, num_hidden1])
    layer4_bias = bias_variable([num_hidden1])

    layer4a_W = weight_variable([num_hidden1, num_hidden2])
    layer4a_bias = bias_variable([num_hidden2])
    #layer 5
    layer5_Ws = [weight_variable([num_hidden2, num_labels]) for _ in xrange(num_digits)]
    layer5_biases = [bias_variable([num_labels]) for _ in xrange(num_digits)]

    #Design model
    def model(data):
        conv1 = tf.nn.relu(tf.nn.conv2d(data, layer1_W, [1,2,2,1], padding='SAME') + layer1_bias)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, layer2_W, [1,2,2,1], padding='SAME') + layer2_bias)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, layer3_W, [1,2,2,1], padding='SAME') + layer3_bias)
        conv3a = tf.nn.relu(tf.nn.conv2d(conv3, layer3a_W, [1,1,1,1], padding='SAME') + layer3a_bias)
        conv3c = tf.nn.relu(tf.nn.conv2d(conv3a, layer3c_W, [1,1,1,1], padding='SAME') + layer3c_bias)
        shape = conv3c.get_shape().as_list()
        reshaped = tf.reshape(conv3c, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden1 = tf.nn.relu(tf.matmul(reshaped, layer4_W) + layer4_bias)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, layer4a_W) + layer4a_bias)
        return [tf.matmul(hidden2, layer5_W) + layer5_bias for layer5_W, layer5_bias in zip(layer5_Ws, layer5_biases)]

    logits = model(tf_train_dataset)
    loss_per_digit = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[i], tf_train_labels[:,i,:])) for i in xrange(num_digits)]
    loss = tf.add_n(loss_per_digit)

    #optimizer
    global_step = tf.Variable(0,trainable=False)
    starter_learning_rate = 0.005
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,5,0.90,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_prediction = tf.transpose(tf.nn.softmax(logits), [1,0,2])
    validation_prediction = tf.transpose(tf.nn.softmax(model(tf_valid_dataset)), [1,0,2])
#    test_prediction = tf.nn.softmax(model(tf_test_dataset))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    

    for step in range(1001):
        batch_data, batch_label = get_train_data('train/',offset,batch_size)
        offset += batch_size
        feed_dict = { tf_train_dataset: batch_data,
                      tf_train_labels : batch_label }
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
        if step%1 == 0:
            print('Minibatch accuracy at step {} : {} and loss is {} with rate {}'.format(step, accuracy(predictions, batch_label),l,learning_rate))
            print('Validation accuracy is {}'.format(accuracy(validation_prediction.eval(), yvalidation)))
                                                             
