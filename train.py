from resnet import *
from datetime import datetime
import time
from input import *
from test import *
import pandas as pd
import glob
import re


class Train(object):
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()

    def placeholders(self):
        '''
        image_placeholder and label_placeholder are for train images and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size, NUM_CLASS])
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])


    def build_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.
        
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        # TRAIN OPERATION
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', self.full_loss)

        # The ema object help calculate the moving average of train loss and train error
        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(self.full_loss, global_step=global_step)

        self.train_op, self.train_ema_op = train_op, 0

    def train(self):
        '''
        This is the main function for training
        '''

        # First load all the paths of image data
        train_image_labels = prepare_data('train', TRAIN_SIZE)
        val_image_labels = prepare_data('val', VAL_SIZE)

        # Build the graph for train
        self.build_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        sess = tf.Session()

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            ckpt_name = model_restore(sess, saver)
            print('Restored from checkpoint... {}'.format(ckpt_name))
        else:
            init = tf.initialize_all_varibles()
            sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        accuracy = [0]*14 

        print('Start training...')
        print('----------------------------')

        for step in range(FLAGS.train_steps):
            indices = get_random_indices(TRAIN_SIZE)
            train_batch_data, train_batch_labels = load_images(indices, train_image_labels, True)

            start_time = time.time()
            _, train_loss_value = sess.run([self.train_op, self.full_loss],
                                {self.image_placeholder: train_batch_data,
                                  self.label_placeholder: train_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time

            # Report to console at specific steps
            if step % FLAGS.report_freq == 0:
                print('{}/{}.. Loss: {}.. Accuracy: {}'.format(step, FLAGS.train_steps, train_loss_value, str(accuracy)))
                step_list.append(step)
                train_error_list.append(train_loss_value)

            # Update learning rate at specific steps
            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)

            # Save checkpoints after certain steps
            if step % 300 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                print('Saving Checkpoint!: {} at {} steps'.format(checkpoint_path, int(ckpt_name)+step))
                saver.save(sess, checkpoint_path, global_step=int(ckpt_name)+step)

                df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                'validation_error': train_error_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')

            """
            if step % 800 == 0:
              indices = get_random_indices(VAL_SIZE, 90)
              val_batch_data, val_batch_labels = load_images(indices, val_image_labels, True)
              net_out, accuracy = test(val_batch_data, val_batch_labels)#, sess, saver)
              df = pd.DataFrame(data={'net_out':net_out, 'labels':val_batch_labels,
                              'accuracy': accuracy})
              df.to_csv(train_dir + FLAGS.version + '_accuracy.csv')
            """

    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


if __name__ == "__main__":
  # Already done!
  #maybe_download_and_extract()
  # Initialize the Train object
  train = Train()
  # Start the training session
  train.train()
