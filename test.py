import tensorflow as tf
from resnet import *
from input import *
from train import *

def test(image_array, image_answers):
    '''
    This function is used to evaluate the test data. Please finish pre-precessing in advance

    :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
    img_depth]
    :return: the softmax probability with shape [num_test_images, num_labels]
    '''
    # update hyperparams
    num_test_images = len(image_array)
    #num_batches = num_test_images // FLAGS.test_batch_size
    #remain_images = num_test_images % FLAGS.test_batch_size
    #print('%i test batches in total...' %num_batches)

    # Create the test image and labels placeholders
    test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

    # Build the test graph
    logits = inference(test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
    predictions = tf.nn.sigmoid(logits)

    #tf.metrics.auc(labels, predictions)

    # Initialize a new session and restore a checkpoint
    saver = tf.train.Saver(tf.all_variables())
    #saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    sess = tf.Session()

    ckpts = glob.glob(FLAGS.ckpt_path)
    latest = max([re.search('\-(\d+)\.', x).groups()[0] for x in ckpts])
    ckpt_path = 'logs_test_110/model.ckpt-' + latest
    saver.restore(sess, FLAGS.ckpt_path)
    #saver.restore(sess, FLAGS.test_ckpt_path)
    print('Model restored from ', FLAGS.test_ckpt_path)

    prediction_array = np.array([]).reshape(-1, NUM_CLASS)
    # Test by batches
    num_batches = 1
    for step in range(num_batches):
        if step % 10 == 0:
            print('%i batches finished!' %step)
        #offset = step * FLAGS.test_batch_size
        #test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

        batch_prediction_array = sess.run(predictions,
                                    feed_dict={test_image_placeholder: image_array})

        prediction_array = np.concatenate((prediction_array, batch_prediction_array))

    # If test_batch_size is not a divisor of num_test_images
    """
    if remain_images != 0:
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
        #predictions = tf.nn.softmax(logits)

        test_image_batch = test_image_array[-remain_images:, ...]

        batch_prediction_array = sess.run(predictions, feed_dict={
            self.test_image_placeholder: test_image_batch})

        prediction_array = np.concatenate((prediction_array, batch_prediction_array))
    """

    return prediction_array

if __name__ == "__main__":
  val_image_labels = prepare_data('val', VAL_SIZE)
  indices = get_random_indices(VAL_SIZE)
  val_batch_data, val_batch_labels = load_images(indices, val_image_labels, True)
  test(val_batch_data, val_batch_labels)
