# -*- coding: utf-8 -*-

import numpy as np
import math
import tensorflow as tf
import os
import time
import operator
import pickle
import random
import sys


tf.app.flags.DEFINE_string('mode', 'train', 'running mode, "train" or "gen"')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_integer('rnn_size', 128, 'rnn hidden size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'rnn num layers.')
tf.app.flags.DEFINE_integer('sequence_len', 50, 'sequence length, input will be formatted to (batch_size,sequence_len)')
tf.app.flags.DEFINE_float('learning_rate', 0.002, 'learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_ratio', 0.97, 'learning rate decay.')
tf.app.flags.DEFINE_float('learning_rate_decay_every', 10, 'epoch numbers that learning rate will decay.')
tf.app.flags.DEFINE_string('input_path', 'data/jinyong.txt', 'input dir.')
tf.app.flags.DEFINE_string('encoding',None, 'data encoding, ascii/utf/etc. default is None (binary)')
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'rnn/gru/lstm')
tf.app.flags.DEFINE_integer('max_epochs', 50000, 'train epochs.')
tf.app.flags.DEFINE_float('validate_set_ratio', 0.05, 'how many data are used as validate set.')
tf.app.flags.DEFINE_integer('print_train_every', 30, 'print train every steps.')
tf.app.flags.DEFINE_integer('print_validate_every', 100, 'print validate every steps.')
tf.app.flags.DEFINE_integer('save_model_every', 100, 'save mode every steps')
tf.app.flags.DEFINE_integer('gen_sentence_len', 100, 'length of sentence generated')
tf.app.flags.DEFINE_integer('random_seed', 123, 'random seed for python/np/tf')
tf.app.flags.DEFINE_string('gpu', '0', '''GPU ID''')

FLAGS=tf.app.flags.FLAGS


def read_dict():
    """
    read dict
    :return: dict {char:idx}, 0-based
    """
    if not os.path.isfile(FLAGS.input_path):
        raise ValueError('can not find input "%s"'%FLAGS.input_path)

    input_dir,input_file=os.path.split(FLAGS.input_path)
    dict_path=os.path.join(input_dir,input_file.split('.')[0]+".dict")

    if not os.path.isfile(dict_path):
        # no vocabulary, create new
        buffer_size = 1000
        f = open(FLAGS.input_path, 'rb') if FLAGS.encoding is None else open(FLAGS.input_path, 'r',encoding=FLAGS.encoding)
        # chars is 1-based index {'a':1,'b':2,...}
        chars = {}
        raw_data = f.read(buffer_size)
        while raw_data:

            for char in raw_data:
                if char not in chars:
                    chars[char] = len(chars)
            raw_data = f.read(buffer_size)
        f.close()

        pickle.dump(chars, open(dict_path, "wb"))
        print('Create vocabulary "%s"' % dict_path)
        return chars
    else:
        with open(dict_path, 'rb') as f:
            chars=pickle.load(f)
        return chars


def read_input(chars):
    """
    read input, encode by dict, split to train set and validate set and save, both of them are 1-D tensor
    :return:
    """
    input_dir, input_file = os.path.split(FLAGS.input_path)
    train_tensor_path=os.path.join(input_dir,input_file.split('.')[0]+".train")
    validate_tensor_path = os.path.join(input_dir, input_file.split('.')[0] + ".validate")
    if not os.path.isfile(train_tensor_path) or not os.path.isfile(validate_tensor_path):
        buffer_size = 1000
        data = []
        f = open(FLAGS.input_path, 'rb') if FLAGS.encoding is None else open(FLAGS.input_path, 'r', encoding=FLAGS.encoding)
        raw_data = f.read(buffer_size)
        while raw_data:
            for char in raw_data:
                char_id=chars[char]
                data.append(char_id)
            raw_data = f.read(buffer_size)
        f.close()
        data=np.array(data)

        def split_data(data):
            """
            split to train set and validate set randomly
            :param data:
            :return: train, validate
            """
            l = len(data)
            vl = math.ceil(l * FLAGS.validate_set_ratio)
            ids = np.random.permutation(l)
            return data[ids[vl:]], data[ids[:vl]]

        train_data,validate_data=split_data(data)
        pickle.dump(train_data, open(train_tensor_path, "wb"))
        pickle.dump(validate_data, open(validate_tensor_path, "wb"))
        print('Create tensor file "%s" and "%s"' % (train_tensor_path,validate_tensor_path))
    else:
        with open(train_tensor_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(validate_tensor_path, 'rb') as f:
            validate_data = pickle.load(f)

    return train_data,validate_data


class DataProvider:

    def __init__(self,data):

        """
        remove tail of data that is smaller than a whole batch,
        :param data:
        :return:
        """

        # remove tail of data, all batches are evenly distributed
        self.batch_num=len(data)//(FLAGS.batch_size*FLAGS.sequence_len)
        data_x=np.array(data[:self.batch_num*(FLAGS.batch_size*FLAGS.sequence_len)])
        data_y=np.zeros(len(data_x),dtype=data_x.dtype)
        data_y[1:]=data_x[0:-1]
        data_y[0]=data_x[-1]
        self.data_x=np.reshape(data_x,[self.batch_num,FLAGS.batch_size,FLAGS.sequence_len])
        self.data_y=np.reshape(data_y,self.data_x.shape)
        """
               self.data_x   self.data_y
               [A,B,C]  [B,C,D]
               [D,E,F]  [E,F,A]
        """

    def next(self, batch_id):
        assert batch_id >= 0 and batch_id < self.batch_num
        return self.data_x[batch_id], self.data_y[batch_id]
    

def rnn_model(cell_type, input_data, output_data, vocab_size, rnn_size, num_layers, batch_size,
              learning_rate):
    """
    construct rnn seq2seq model.
    :param cell_type: cell_type class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    end_points = {}

    if cell_type== 'rnn':
        cell_fun=tf.nn.rnn_cell.BasicRNNCell
    elif cell_type == 'gru':
        cell_fun=tf.nn.rnn_cell.GRUCell
    elif cell_type == 'lstm':
        cell_fun =tf.nn.rnn_cell.LSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device("/cpu:0"),tf.name_scope("hidden"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        tf.summary.histogram("embedding", embedding)

    # [batch_size, ?, rnn_size] = [64, ?, 128]
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size)
        # should be [?, vocab_size+1]

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size+1]
        total_loss = tf.reduce_mean(loss)

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points


class RNN:
    def __init__(self):
        self.chars=read_dict()
        self.vocab_size = len(self.chars)
        print('Find %d word(s) in vocabulary'%self.vocab_size)

        _, input_file = os.path.split(FLAGS.input_path)
        model_root = "model"
        if not os.path.exists(model_root):
            os.mkdir(model_root)
            print("create model directory %s" % model_root)

        self.input_prefix=input_file.split('.')[0]

        self.model_dir = os.path.join(model_root, self.input_prefix)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
            print("create model directory %s" % self.model_dir)

        log_root = "logs"
        if not os.path.exists(log_root):
            os.mkdir(log_root)
            print("create log directory %s" % log_root)

        self.log_dir = os.path.join(log_root, self.input_prefix)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            print("create log directory %s" % self.log_dir)

    def train(self):
        train_data,validate_data=read_input(self.chars)

        train_dp, validate_dp = DataProvider(train_data), DataProvider(validate_data)
        print('Find %d records(%d batches) in train set and %d records(%d batchs) in validate set' % (len(train_data),train_dp.batch_num,len(validate_data),validate_dp.batch_num))
        print("## train_dp.data_x[0][0][:10]: %s" % str(train_dp.data_x[0][0][:10]))
        print("## train_dp.data_y[0][0][-10:]: %s" % str(train_dp.data_y[0][0][-10:]))
        print("## validate_dp.data_x[0][0][:10]: %s" % str(validate_dp.data_x[0][0][:10]))
        print("## validate_dp.data_y[0][0][-10:]: %s" % str(validate_dp.data_y[0][0][-10:]))


        input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None], 'input_data')
        output_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None], 'output_data')
        endpoints = rnn_model(cell_type=FLAGS.cell_type, input_data=input_data, output_data=output_data, vocab_size=self.vocab_size, rnn_size=FLAGS.rnn_size,
                              num_layers=FLAGS.num_layers,batch_size=FLAGS.batch_size,learning_rate=FLAGS.learning_rate)
        merge_summary_op = tf.summary.merge_all()
        global_step=0
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()]))
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print('Load model "%s", go on' % checkpoint, flush=True)
            else:
                print('Can not find model from "%s", restart training' % self.model_dir, flush=True)
    
            train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"), sess.graph)
            validate_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "validate"), sess.graph)
    
            # max train step
            max_step = FLAGS.max_epochs*train_dp.batch_num
           
            try:
                while global_step<=max_step:
                    validate_batch_id=0
                    epoch_id = global_step // train_dp.batch_num
                    batch_id = global_step % train_dp.batch_num
                    train_x, train_y = train_dp.next(batch_id)
                    if global_step%FLAGS.print_train_every == 0 and global_step%FLAGS.print_validate_every==0:
                        # train & validate
                        validate_x,validate_y=validate_dp.next(validate_batch_id%validate_dp.batch_num)
                        _,last_state,train_total_loss,train_summary\
                            = sess.run([endpoints['train_op'], endpoints['last_state'],endpoints['total_loss'],
                                            merge_summary_op],feed_dict={input_data:train_x,output_data:train_y})
                        
                        validate_total_loss,validate_last_state,validate_summary = sess.run([endpoints['total_loss'], endpoints['last_state'],merge_summary_op],
                                                            feed_dict={input_data: validate_x, output_data: validate_y})
                        validate_batch_id += 1
                        train_writer.add_summary(train_summary, global_step)
                        train_writer.flush()
    
                        validate_writer.add_summary(validate_summary,global_step)
                        validate_writer.flush()
                        print('[%s] Global Step %d, Epoch %d, Batch %d, Train Loss=%.8f, Learning Rate=%.8f, Validate Loss=%.8f'%
                              (time.strftime('%Y-%m-%d %H:%M:%S'),global_step,epoch_id,batch_id,train_total_loss,FLAGS.learning_rate,validate_total_loss),flush=True)
                    elif global_step%FLAGS.print_train_every == 0:
                        # train only
                        _,last_state, train_total_loss, train_summary \
                            = sess.run([endpoints['train_op'], endpoints['last_state'], endpoints['total_loss'],
                                             merge_summary_op],
                                            feed_dict={input_data: train_x, output_data: train_y})
                        train_writer.add_summary(train_summary, global_step)
                        train_writer.flush()
                        print('[%s] Global Step %d, Epoch %d, Batch %d, Train Loss=%.8f, Learning Rate=%.8f' % (
                        time.strftime('%Y-%m-%d %H:%M:%S'), global_step,epoch_id, batch_id, train_total_loss, FLAGS.learning_rate),flush=True)
                    elif global_step%FLAGS.print_validate_every==0:
                        # validate only
                        validate_x, validate_y = validate_dp.next(validate_batch_id % validate_dp.batch_num)
                        _,last_state, train_total_loss= sess.run(
                            [endpoints['train_op'], endpoints['last_state'], merge_summary_op],
                            feed_dict={input_data: train_x, output_data: train_y})
                        validate_total_loss, validate_summary = sess.run(
                            [endpoints['total_loss'], merge_summary_op],
                            feed_dict={input_data: validate_x, output_data: validate_y})
                        validate_batch_id += 1
                        validate_writer.add_summary(validate_summary, global_step)
                        validate_writer.flush()
                        print('[%s] Global Step %d, Epoch %d, Batch %d, Validate Loss=%.8f' % (time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                               global_step,epoch_id, batch_id, validate_total_loss),flush=True)
                    else:
                        # nothing
                        _,last_state = sess.run(
                            [endpoints['train_op'], endpoints['last_state']],
                            feed_dict={input_data: train_x, output_data: train_y})
    
                    if global_step % FLAGS.save_model_every == 0:
                        # 保存模型
                        file_path=os.path.join(self.model_dir,self.input_prefix)
                        saver.save(sess, file_path, global_step=global_step)
                        print('[%s] Save model %s-%d' % (time.strftime('%Y-%m-%d %H:%M:%S'), file_path,global_step),
                              flush=True)

                    # global_step has been increased by optimizer
                    global_step+=1
    
                # save when exit
                saver.save(sess, file_path, global_step=global_step)
            except KeyboardInterrupt as e:
                print('Meet KeyboardInterrupt %s'%e)
                file_path = os.path.join(self.model_dir, self.input_prefix)
                saver.save(sess, file_path, global_step=global_step)
                print('[%s] Save model %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), file_path),
                      flush=True)

    def gen(self):
        batch_size=1
        seq_len=1
        input_data = tf.placeholder(tf.int32, [batch_size, seq_len], 'input_data')

        endpoints = rnn_model(cell_type=FLAGS.cell_type, input_data=input_data, output_data=None,
                              vocab_size=self.vocab_size, rnn_size=FLAGS.rnn_size,
                              num_layers=FLAGS.num_layers,batch_size=batch_size, learning_rate=FLAGS.learning_rate)

        # vocabs is list of all chars
        vocabs = list(map(lambda x: x[0], sorted(self.chars.items(), key=operator.itemgetter(1))))

        def to_vocab(predict):
            predict = predict[0]
            sample = np.random.choice(np.arange(len(predict)), p=predict)
            return vocabs[sample]

        def prediction_to_vocab(prediction):
            vocab = to_vocab(prediction[0])
            vocab_id = self.chars[vocab]
            return vocab_id, vocab

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print('Load model from "%s", gen sentence:' % checkpoint, flush=True)
            else:
                print('Can not find model from "%s", exit !' % self.model_dir, flush=True)
                return

            while True:

                input('## print any key to compose new sentence')

                # pick first word randomly
                word = vocabs[np.random.randint(len(vocabs))]
                output = [word]

                x = np.array([[self.chars[word]]])
                [predict, last_state] = sess.run([endpoints['prediction'], endpoints['last_state']],
                                                 feed_dict={input_data: x})
                # second word
                word = to_vocab(predict)

                i = 1
                while i < FLAGS.gen_sentence_len:
                    output.append(word)
                    i += 1
                    x = np.zeros((1, 1))
                    x[0, 0] = self.chars[word]
                    [predict, last_state] = sess.run([endpoints['prediction'], endpoints['last_state']],
                                                     feed_dict={input_data: x, endpoints['initial_state']: last_state})
                    word = to_vocab(predict)

                if isinstance(output[0],int):
                    sys.stdout.buffer.write(
                        b'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
                    sys.stdout.buffer.write(bytes(output))
                    sys.stdout.buffer.write(
                        b'------------------------------------------------------------------------------\n')
                    sys.stdout.buffer.write(str(output).encode())
                    sys.stdout.buffer.write(b'\n')
                    sys.stdout.buffer.write(
                        b'==============================================================================\n')
                    sys.stdout.buffer.write(b'\n')

                    sys.stdout.flush()
                else:
                    print("".join(output), end='\n', flush=True)



if __name__ == '__main__':
    def main(_):
        print('='*100)
        print('[FLAGS]')
        for k,v in sorted(FLAGS.flag_values_dict().items(),key=operator.itemgetter(0)):
            if k in ['h','help','helpfull','helpshort']:
                continue

            if isinstance(v,str):
                print('%s = "%s"' % (k, v))
            else:
                print('%s = %s' % (k, v))
        print('=' * 100, flush=True)
        np.random.seed(FLAGS.random_seed)
        random.seed(FLAGS.random_seed)
        tf.set_random_seed(FLAGS.random_seed)
        rnn=RNN()

        if FLAGS.mode == 'train':
            rnn.train()
        elif FLAGS.mode == 'gen':
            rnn.gen()


    if __name__ == '__main__':
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu  # set GPU visibility in multiple-GPU environment
        tf.app.run()