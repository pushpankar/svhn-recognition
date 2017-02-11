import tensorflow as tf
import numpy as np
from svhn_data import get_train_data, get_camera_images


def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='Weight')


def bias_var(shape):
    return tf.Variable(tf.constant(1.0, shape=shape), name='bias')


def accuracy(pred, labels):
    return (100.0 * np.sum(
        np.argmax(pred, 2) == np.argmax(labels, 2)) / pred.shape[0])/6


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.histogram('histogram', var)


def conv2d(data, wt, bias, max_pool=True):
    stride = [1, 1, 1, 1]
    variable_summaries(wt)
    variable_summaries(bias)
    conv = tf.nn.relu(tf.nn.conv2d(data, wt, stride, padding='SAME', name='convolution') + bias,
                      name='relu')
    if max_pool:
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.dropout(conv, 0.80, name='dropout')


offset = 0
image_height = 32
image_width = 128
num_channels = 1
num_labels = 11
num_digits = 6

batch_size = 64
patch_size = 5
depth = 32
num_hidden1 = 1024
reg_hidden1 = 1024
reg_hidden2 = 512

Xvalid, yvalid, bbox_valid = get_train_data('train/', offset, batch_size//2)
offset += batch_size//2
Xtest, ytest, bbox_test = get_train_data('train/', offset, batch_size//2)
offset += batch_size//2

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
    layer2_W = weight_var([patch_size, patch_size, depth, depth*2])
    layer2_bias = bias_var([depth*2])

    # layer3
    layer3_W = weight_var([patch_size, patch_size, depth*2, depth*2])
    layer3_bias = bias_var([depth*2])

    # layer4
    layer4_W = weight_var([patch_size, patch_size, depth*2, depth*2])
    layer4_bias = bias_var([depth*2])

    # layer5
    layer5_W = weight_var([patch_size, patch_size, depth*2, depth*2])
    layer5_bias = bias_var([depth*2])

    # convolution layer 6
    layer6_W = weight_var([patch_size, patch_size, depth*2, depth*4])
    layer6_bias = bias_var([depth*4])

    # convolution layer 7
    layer7_W = weight_var([patch_size, patch_size, depth*4, depth*4])
    layer7_bias = bias_var([depth*4])

    # Regression head
    fc1_reg_W = weight_var([image_height//16*image_width//16*depth*4, reg_hidden1])
    fc1_reg_bias = bias_var([reg_hidden1])

    fc2_reg_W = weight_var([reg_hidden1, reg_hidden2])
    fc2_reg_bias = bias_var([reg_hidden2])

    fc3_reg_Ws = [weight_var([reg_hidden2, 4]) for _ in range(num_digits)]
    fc3_reg_biases = [bias_var([4]) for _ in range(num_digits)]

    # Classification head
    c1_W = weight_var([image_height//16*image_width//16*depth*4, num_hidden1])
    c1_bias = bias_var([num_hidden1])

    c3_Ws = [weight_var([num_hidden1, num_labels])
             for _ in range(num_digits)]
    c3_biases = [bias_var([num_labels]) for _ in range(num_digits)]

    # Design model
    def model(data):
        with tf.name_scope('conv1'):
            conv1 = conv2d(data, layer1_W, layer1_bias)
        with tf.name_scope('conv2'):
            conv2 = conv2d(conv1, layer2_W, layer2_bias)
        with tf.name_scope('conv3'):
            conv3 = conv2d(conv2, layer3_W, layer3_bias)
        with tf.name_scope('conv4'):
            conv4 = conv2d(conv3, layer4_W, layer4_bias, max_pool=False)
        with tf.name_scope('conv5'):
            conv5 = conv2d(conv4, layer5_W, layer5_bias, max_pool=False)
        with tf.name_scope('conv6'):
            conv6 = conv2d(conv5, layer6_W, layer6_bias)
        with tf.name_scope('conv7'):
            conv7 = conv2d(conv6, layer7_W, layer7_bias, max_pool=False)
        with tf.name_scope('reshape'):
            shape = conv7.get_shape().as_list()
            reshaped = tf.reshape(conv7, [shape[0], shape[1] * shape[2] * shape[3]])

        # Regression head for bounding box prediction
        with tf.name_scope('reg1'):
            reg1 = tf.nn.relu(tf.matmul(reshaped, fc1_reg_W) + fc1_reg_bias)
        with tf.name_scope('reg2'):
            reg2 = tf.nn.relu(tf.matmul(reg1, fc2_reg_W) + fc2_reg_bias)
        with tf.name_scope('reg_list'):
            bbox_pred = [tf.matmul(reg2, fc3_reg_W) + fc3_reg_bias
                         for fc3_reg_W, fc3_reg_bias in zip(fc3_reg_Ws, fc3_reg_biases)]
        with tf.name_scope('transpose'):
            bbox_pred = tf.transpose(tf.pack(bbox_pred), [1, 0, 2])

        # classification head
        with tf.name_scope('classify1'):
            hidden1 = tf.nn.relu(tf.matmul(reshaped, c1_W) + c1_bias)
        with tf.name_scope('classify_list'):
            logits = tf.pack([tf.matmul(hidden1, c3_W) + c3_bias
                              for c3_W, c3_bias in zip(c3_Ws, c3_biases)])
            logits = tf.transpose(logits, [1, 0, 2])
        return logits, bbox_pred

    logits, train_bbox_pred = model(tf_train_dataset)
    loss_per_digit = [tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits[:, i, :], tf_train_labels[:, i, :]))
                      for i in range(num_digits)]
    with tf.name_scope('loss'):
        loss = tf.add_n(loss_per_digit, name='loss')
        bbox_loss = tf.nn.l2_loss(train_bbox_pred - tf_train_bbox, name='reg_loss')

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('bbox_loss', bbox_loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('log', graph)

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.005
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step, 5, 0.90,
                                               staircase=True)
    total_loss = bbox_loss + loss
    optimizer = tf.train.AdagradOptimizer(
        learning_rate).minimize(total_loss, global_step=global_step)
    # bbox_optimizer = tf.train.AdagradOptimizer(
    #    learning_rate).minimize(bbox_loss, global_step=global_step)

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
        batch_data, batch_label, bbox = get_train_data('train/', offset, batch_size//2)
        offset += batch_size//2
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_label,
                     tf_train_bbox: bbox}
        _, l, predictions, bbox_cost, summary = session.run(
            [optimizer, loss, train_pred, bbox_loss, merged],
            feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        if step % 10 == 0:
            print('Minibatch accuracy at step {} : {} and loss is {}'.format(
                step, accuracy(predictions, batch_label), l))
            print('Validation accuracy is {}'.format(
                accuracy(valid_pred.eval(), yvalid)))
            print('Regression loss is {} is {}'.format(step, bbox_cost))

    print('Test accuracy is {}'.format(accuracy(test_pred.eval(), ytest)))
    print('Camera result is {} and bbox is {}'.format(
        np.argmax(camera_image_pred.eval(), 2), camera_bbox_pred.eval()))
