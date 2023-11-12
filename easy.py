
import datetime
import logging
import os
import math
import time
from PIL import Image
import numpy as np
import tensorflow as tf

import orcmodel
import utils

FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    model = orcmodel.LSTMOCR(mode)
    model.build_graph()

    print('loading train data, please wait---------------------')
    train_feeder = utils.DataIterator(data_dir=train_dir) 
    print('get image: ', train_feeder.size)

    print('loading validation data, please wait---------------------')
    val_feeder = utils.DataIterator(data_dir=val_dir)
    print('get image: ', val_feeder.size)

    num_train_samples = train_feeder.size  
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size) 

    num_val_samples = val_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size) 
    shuffle_idx_val = np.random.permutation(num_val_samples)

    with tf.device('/cpu:0'):

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph) 

            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:

                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))

            print('=============================begin training=============================')
            for cur_epoch in range(FLAGS.num_epochs):
                shuffle_idx = np.random.permutation(num_train_samples)
                train_cost = 0
                start_time = time.time()
                batch_time = time.time()

                for cur_batch in range(num_batches_per_epoch):
                    if (cur_batch + 1) % 100 == 0:
                        print('batch', cur_batch, ': time', time.time() - batch_time)
                    batch_time = time.time()
                    
                    indexs = [shuffle_idx[i % num_train_samples] for i in
                              range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                    batch_inputs, batch_seq_len, batch_labels = \
                        train_feeder.input_index_generate_batch(indexs)
                    
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels,
                            model.seq_len: batch_seq_len}

                    summary_str, batch_cost, step, _ = \
                        sess.run([model.merged_summay, model.cost, model.global_step,
                                  model.train_op], feed)
                    
                    train_cost += batch_cost * FLAGS.batch_size

                    train_writer.add_summary(summary_str, step) 

           
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        logger.info('save the checkpoint of{0}', format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                                   global_step=step)

                    if step % FLAGS.validation_steps == 0:
                        acc_batch_total = 0
                        lastbatch_err = 0
                        lr = 0
                        for j in range(num_batches_per_epoch_val):
                            indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                          range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                            val_inputs, val_seq_len, val_labels = \
                                val_feeder.input_index_generate_batch(indexs_val)
                            val_feed = {model.inputs: val_inputs,
                                        model.labels: val_labels,
                                        model.seq_len: val_seq_len}

                            dense_decoded, lr = sess.run([model.dense_decoded, model.lrn_rate], val_feed)

                            ori_labels = val_feeder.the_label(indexs_val)
                            
                            acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                             ignore_value=-1, isPrint=True)
                            acc_batch_total += acc
                        avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)
                        now = datetime.datetime.now()
                        log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                              "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                              "lastbatch_cost = {:f}, time = {:.3f},lr={:.8f}"
                        print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                         cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
                                         batch_cost, time.time() - start_time, lr))
def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO) 
    tf.app.run() 