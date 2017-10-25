import tensorflow as tf
from resnet import *
from input import *
#from train import *
import glob
import re

def model_restore(sess, saver):
    fixed_path = FLAGS.ckpt_path.split('-')[0] + "*"
    ckpts = glob.glob(fixed_path)
    latest = str(max([int(re.search('\-(\d+)\.', x).groups()[0]) for x in ckpts]))
    ckpt_path = 'logs_test_110/model.ckpt-' + latest
    saver.restore(sess, ckpt_path)
    return latest

def test(image_array, image_answers):
    '''
    This function is used to evaluate the test data. Please finish pre-precessing in advance

    :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
    img_depth]
    :return: the softmax probability with shape [num_test_images, num_labels]
    '''
    # update hyperparams
    num_test_images = len(image_array)
    num_batches = num_test_images // FLAGS.test_batch_size
    #remain_images = num_test_images % FLAGS.test_batch_size
    #print('%i test batches in total...' %num_batches)

    #if sess is None or saver is None:
    # Create the test image and labels placeholders
    test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

    # Build the test graph
    logits = inference(test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
    predictions = tf.nn.sigmoid(logits)

    # Initialize a new session and restore a checkpoint
    #saver = tf.train.Saver(tf.all_variables())
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    #init = tf.initialize_all_variables()
    sess = tf.Session()

    #tf.metrics.auc(labels, predictions)
    ckpt_pth = model_restore(sess, saver)
    print('Model restored from', ckpt_pth)

    prediction_out = np.array([]).reshape(-1, NUM_CLASS)
    # Test by batches
    #num_batches = 30
    for step in range(num_batches):
        if step % 10 == 0:
            print('%i batches finished!' %step)
        batch_image_array = image_array[step*3:step*3+3]
        batch_prediction_out = sess.run(predictions,
                                    feed_dict={test_image_placeholder: batch_image_array})

        #for prediction in batch_prediction_logits:
        prediction_out = np.concatenate((prediction_out, batch_prediction_out))

    #odds = np.exp(prediction_logits)
    #prediction_array = odds / (1 + odds)
    means = np.mean(prediction_out, 0)
    correct = [0]*NUM_CLASS
    for idx in range(NUM_CLASS):
      for ele in range(num_test_images):
        if prediction_out[ele][idx] > means[idx] and image_answers[ele][idx]==1.0:
          correct[idx] += 1
        if prediction_out[ele][idx] < means[idx] and image_answers[ele][idx]==0.0:
          correct[idx] += 1
    correct = [val/num_test_images for val in correct]
    #print(correct)
    return prediction_out, correct 

if __name__ == "__main__":
  val_image_labels = prepare_data('val', VAL_SIZE)
  indices = get_random_indices(VAL_SIZE, 30) #10000)
  val_batch_data, val_batch_labels = load_images(indices, val_image_labels, True)
  po, co = test(val_batch_data, val_batch_labels)
  print(po.shape)
  print(val_batch_labels.shape)
  with open('out.csv', 'a') as f:
    for i in range(len(po)):
      f.write(str(po[i]))
      f.write(',')
      f.write(str(val_batch_labels[i]))
      f.write('\n')
