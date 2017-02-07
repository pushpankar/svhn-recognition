import tensorflow as tf
import numpy as np
from svhn_data import get_train_data, get_camera_images


def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_var(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))


def accuracy(pred, labels):
    return (100.0 * np.sum(
        np.argmax(pred, 2) == np.argmax(labels, 2)) / pred.shape[0])/6


def conv2d(data, wt, bias, stride=[1, 2, 2, 1]):
    return tf.nn.relu(tf.nn.conv2d(data, wt, stride, padding='SAME') + bias)


offset = 0
image_height = 32
image_width = 128
num_channels = 1
num_labels = 11
num_digits = 6

batch_size = 64
patch_size = 5
depth = 128
num_hidden1 = 2048
num_hidden2 = 512
reg_hidden1 = 2048
reg_hidden2 = 1024
reg_hidden3 = 512

Xvalid, yvalid, bbox_valid = get_train_data('train/', offset, batch_size)
offset += batch_size
Xtest, ytest, bbox_test = get_train_data('train/', offset, batch_size)
offset += batch_size


# print dimensions
print("Xvalid shape is {} and yvalid shape is {} and bbox is {}"
      .format(Xvalid.shape, yvalid.shape, bbox_valid.shape))
print('y is \n{}\nbbox is \n{}'
      .format(yvalid[0], bbox_valid[0]))

# build a graph
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size,
                           image_width, image_height, num_channels))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_digits, num_labels))
    tf_valid_dataset = tf.cast(tf.constant(Xvalid), tf.float32)
    tf_train_bbox = tf.placeholder(tf.float32,
                                   shape=(batch_size, num_digits, 4))
    tf_test_dataset = tf.cast(tf.constant(Xtest), tf.float32)
    tf_camera_data = tf.cast(tf.constant(get_camera_images()), tf.float32)

    # Create variables
    # convolutions layer 1
    layer1_W = weight_var([patch_size, patch_size, num_channels, depth])
    layer1_bias = bias_var([depth])

    # layer2
    layer2_W = weight_var([patch_size, patch_size, depth, depth])
    layer2_bias = bias_var([depth])

    # layer3
    layer3_W = weight_var([patch_size, patch_size, depth, depth])
    layer3_bias = bias_var([depth])

    # layer4
    layer4_W = weight_var([patch_size, patch_size, depth, depth])
    layer4_bias = bias_var([depth])

    # layer5
    layer5_W = weight_var([patch_size, patch_size, depth, depth])
    layer5_bias = bias_var([depth])

    # Regression head
    fc1_Reg_W = weight_var([image_height//8*image_width//8*depth, reg_hidden1])
    fc1_reg_bias = bias_var([reg_hidden1])

    fc2_reg_W = weight_var([reg_hidden1, reg_hidden2])
    fc2_reg_bias = bias_var([reg_hidden2])

    fc3_reg_W = weight_var([reg_hidden2, reg_hidden3])
    fc3_reg_bias = bias_var([reg_hidden3])

    fc4_reg_Ws = [weight_var([reg_hidden3, 4]) for _ in range(num_digits)]
    fc4_reg_biases = [bias_var([4]) for _ in range(num_digits)]

    # Classification head
    # layer 4
    c1_W = weight_var([image_height//8*image_width//8*depth, num_hidden1])
    c1_bias = bias_var([num_hidden1])

    c2_W = weight_var([num_hidden1, num_hidden2])
    c2_bias = bias_var([num_hidden2])

    # layer 5
    c3_Ws = [weight_var([num_hidden2, num_labels])
             for _ in range(num_digits)]
    c3_biases = [bias_var([num_labels]) for _ in range(num_digits)]

    # Design model
    def model(data):
        conv1 = conv2d(data, layer1_W, layer1_bias)
        conv2 = conv2d(conv1, layer2_W, layer2_bias)
        conv3 = conv2d(conv2, layer3_W, layer3_bias)
        conv4 = conv2d(conv3, layer4_W, layer4_bias, stride=[1, 1, 1, 1])
        conv5 = conv2d(conv4, layer5_W, layer5_bias, stride=[1, 1, 1, 1])
        shape = conv5.get_shape().as_list()
        reshaped = tf.reshape(conv5, [shape[0], shape[1] * shape[2] * shape[3]])

        # Regression head for bounding box prediction
        reg1 = tf.nn.relu(tf.matmul(reshaped, fc1_Reg_W) + fc1_reg_bias)
        reg2 = tf.nn.relu(tf.matmul(reg1, fc2_reg_W) + fc2_reg_bias)
        reg3 = tf.nn.relu(tf.matmul(reg2, fc3_reg_W) + fc3_reg_bias)
        bbox_pred = [tf.matmul(reg3, fc4_reg_W) + fc4_reg_bias
                     for fc4_reg_W, fc4_reg_bias in zip(fc4_reg_Ws, fc4_reg_biases)]
        bbox_pred = tf.transpose(tf.pack(bbox_pred), [1, 0, 2])

        # classification head
        hidden1 = tf.nn.relu(tf.matmul(reshaped, c1_W) + c1_bias)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, c2_W) + c2_bias)
        logits = tf.pack([tf.matmul(hidden2, c3_W) + c3_bias
                          for c3_W, c3_bias in zip(c3_Ws, c3_biases)])
        logits = tf.transpose(logits, [1, 0, 2])
        return logits, bbox_pred

    logits, train_bbox_pred = model(tf_train_dataset)
    loss_per_digit = [tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits[:, i, :], tf_train_labels[:, i, :]))
                      for i in range(num_digits)]
    loss = tf.add_n(loss_per_digit)
    bbox_loss = tf.nn.l2_loss(train_bbox_pred - tf_train_bbox)

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.005
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step, 5, 0.90,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate).minimize(loss, global_step=global_step)
    bbox_optimizer = tf.train.AdamOptimizer(
        learning_rate).minimize(bbox_loss, global_step=global_step)

    # Predictions
    valid_logits, valid_bbox_pred = model(tf_valid_dataset)
    test_logits, test_bbox_pred = model(tf_test_dataset)
    camera_logits, camera_bbox_pred = model(tf_camera_data)
    train_pred = tf.nn.softmax(logits)
    valid_pred = tf.nn.softmax(valid_logits)
    test_pred = tf.nn.softmax(test_logits)
    camera_image_pred = tf.nn.softmax(camera_logits)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(301):
        batch_data, batch_label, bbox = get_train_data('train/', offset, batch_size)
        offset += batch_size
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_label,
                     tf_train_bbox: bbox}
        _, l, predictions, _, bbox_cost = session.run(
            [optimizer, loss, train_pred, bbox_optimizer, bbox_loss],
            feed_dict=feed_dict)
        if step % 25 == 0:
            print('Minibatch accuracy at step {} : {} and loss is {}'.format(
                step, accuracy(predictions, batch_label), l))
            print('Validation accuracy is {}'.format(
                accuracy(valid_pred.eval(), yvalid)))
            print('Regression loss is {} is {}'.format(step, bbox_cost))

    print('Test accuracy is {}'.format(accuracy(test_pred.eval(), ytest)))
    print('Camera result is {} and bbox is {}'.format(
        np.argmax(camera_image_pred.eval(), 2),
        np.argmax(camera_bbox_pred.eval(), 2)))
