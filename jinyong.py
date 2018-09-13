# -*- coding: utf-8 -*-

import collections
import numpy as np
import tensorflow as tf
import os
import time


tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_integer('rnn_size', 128, 'rnn hidden size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'rnn num layers.')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('./model/jinyong'), 'model save path.')
tf.app.flags.DEFINE_string('log_dir', os.path.abspath('./logs/jinyong'), 'logs save path.')
tf.app.flags.DEFINE_string('cell_type', 'rnn', 'rnn/gru/lstm')
tf.app.flags.DEFINE_string('model_prefix', 'jinyong', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 1, 'train how many epochs.')
tf.app.flags.DEFINE_integer('training_echo_interval', 1, 'echo train logs interval.')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 2, 'save model interval during training.')
tf.app.flags.DEFINE_integer('gen_line_interval', 4, 'generate demo line interval.')
tf.app.flags.DEFINE_string('mode','train' , 'train/gen, train model or gen poem use model')
tf.app.flags.DEFINE_string('cuda_visible_devices', '0', '''[Train] visible GPU ''')

FLAGS=tf.app.flags.FLAGS
start_token='B'
end_token='E'
data_path='data/jinyong'
dict_path=data_path+'/dict.txt'
train_path=data_path+'/train.tfrecords'
# 开始
SOS='SOS'
# 结束
EOS='EOS'


def print_args():
    print("系统变量")
    print("mode = '%s'" % FLAGS.mode)
    print("batch_size = %d" % FLAGS.batch_size)
    print("rnn_size = %d" % FLAGS.rnn_size)
    print("learning_rate = %f" % FLAGS.learning_rate)
    print("model_dir = '%s'" % FLAGS.model_dir)
    print("log_dir = '%s'" % FLAGS.log_dir)
    print("train_path = '%s'" % train_path)
    print("log_dir = '%s'" % FLAGS.log_dir)
    print("cell_type = '%s'" % FLAGS.cell_type)
    print("log_dir = '%s'" % FLAGS.log_dir)
    print("model_prefix = '%s'" % FLAGS.model_prefix)
    print("cuda_visible_devices = '%s'" % FLAGS.cuda_visible_devices)
    print("epochs = %d" % FLAGS.epochs)
    print("training_echo_interval = %d" % FLAGS.training_echo_interval)
    print("gen_line_interval = %d" % FLAGS.gen_line_interval)
    print("save_checkpoints_steps = %d" % FLAGS.save_checkpoints_steps)
    print(flush=True)

def process_corpus(src_path):
    """
    处理预料库
    :param src_path:
    :param batch_path:
    :return:
    """
    lines=[]
    for name in os.listdir(src_path):
        # 只处理原始文本
        if name.startswith('u') and name.endswith('.txt'):
            with open(os.path.join(src_path,name),'r',encoding='utf8') as f:
                lines+=f.readlines()

    lines =list([line.strip() for line in lines])
    lines.sort(key=len)
    lines=list(filter(lambda x:len(x)>=3,lines ))
    all_words = [word for line in lines for word in line ]
    # 给SOS EOS 编号
    all_words +=[SOS,EOS]
    counter = collections.Counter(all_words)
    counter_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    words, _ = zip(*counter_pairs)
    word_idx_map = dict(zip(words, range(len(words))))
    sos_idx,eos_idx=word_idx_map[SOS],word_idx_map[EOS]
    sequences = [list(map(lambda word: word_idx_map[word], line)) for line in lines]

    print(u"【搜集数据】共%d行，其中最短行%d字，最长行%d字，共%d个字" % (len(lines),len(lines[0]),len(lines[-1]),len(word_idx_map)), flush=True)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("【新建目录】%s"%data_path)

    # write dict
    with open(dict_path, 'w', encoding='utf') as f:
        for e in word_idx_map:
            f.write('%d:%s\n' % (word_idx_map[e], e))
        print("【创建词典文件】%s" % dict_path)

    # write data
    def _int64list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    with tf.python_io.TFRecordWriter(train_path) as writer:
        for seq in sequences:
            feature={
                'inputs':_int64list_feature([sos_idx]+seq), # 输入需要在前面添加SOS
                'targets': _int64list_feature(seq+[eos_idx])  # 输出需要在后面添加EOS
            }
            # example对象对输入和输出数据进行封装
            example= tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString()) # 序列化为字符串
    print("【创建训练数据】%s" % train_path)


def _read_dict():
    # 读取words
    with open(dict_path, 'r', encoding='utf') as f:
        words = []
        for line in f.read().split('\n')[:-1]:
            pair = line.split(':')
            words.append(pair[1])
    return words

def _read_train_data(batch_size):
    """
    读取训练数据
    :param batch_size:批量大小
    :return: inputs,targets, tensor
    """

    # 读取tf records
    filenames=[train_path]
    dataset=tf.data.TFRecordDataset(filenames)

    def _parse_function(example_proto):
        keys_to_features={
            'inputs':tf.VarLenFeature(tf.int64), # 序列是变长的，需要用tf.VarLenFeature
            'targets': tf.VarLenFeature(tf.int64)
        }
        parsed_features=tf.parse_single_example(example_proto,keys_to_features)
        inputs = tf.sparse_tensor_to_dense(parsed_features['inputs'])
        inputs = tf.cast(inputs,tf.int32) # 转换成tf.int32，因为后面logit是tf.int32
        targets = tf.sparse_tensor_to_dense(parsed_features['targets'])
        targets = tf.cast(targets, tf.int32)
        return inputs,targets

    dataset=dataset.map(_parse_function)
    dataset=dataset.repeat(FLAGS.epochs)
    dataset=dataset.padded_batch(batch_size,padded_shapes=([None],[None])) # 动态pad,这里返回两个值都是变长，所以要两个None
    iterator=dataset.make_one_shot_iterator()
    inputs,targets=iterator.get_next()
    return inputs,targets

def rnn_model(cell_type,input_data,output_data,vocab_size,rnn_size,num_layers,learning_rate):
    """
    Create a RNN model
    :param cell_type: cell type, one of 'rnn'/'gru'/'lstm'
    :param input_data:
    :param output_data: output_data tensor of shape (batch,time) is used as labels to calculate loss, None in predictions
    :param vocab_size: vocabulary size,
    :param rnn_size: the depth of internal neural network
    :param num_layers:
    :param learning_rate:
    :return: endpoints
    """
    if cell_type=='rnn':
        cell=tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)
    elif cell_type == 'gru':
        cell=tf.nn.rnn_cell.GRUCell(num_units=rnn_size)
    elif cell_type == 'lstm':
        cell =tf.nn.rnn_cell.LSTMCell(num_units=rnn_size,state_is_tuple=True)
    else:
        raise ValueError('wrong cell_type "%s"'%cell_type)

    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    end_points={}
    initial_state=cell.zero_state(FLAGS.batch_size,tf.float32) if output_data is not None else cell.zero_state(1,tf.float32)
    # be equivalent to `tf.matmul(input_data_one_hot, embedding)` but avoid matrix multiply
    embedding=tf.get_variable('embedding',initializer=tf.random_uniform([vocab_size,rnn_size],-1.0,1.0))
    inputs=tf.nn.embedding_lookup(embedding,input_data)

    # [batch_size,?, rnn_size]=[64,?,128]
    outputs,last_state=tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)
    output=tf.reshape(outputs,[-1,rnn_size])

    output_weights= tf.get_variable('weights',initializer=tf.truncated_normal([rnn_size,vocab_size]))
    output_bias=tf.get_variable('bias',initializer=tf.zeros([vocab_size]))
    # [?,vocab_size]
    output_logits =tf.nn.bias_add(tf.matmul(output,output_weights),bias=output_bias)
    end_points['initial_state'] = initial_state
    end_points['output'] = output
    end_points['output_logits'] = output_logits
    end_points['last_state'] = last_state

    prediction = tf.nn.softmax(output_logits)
    end_points['last_state'] = last_state
    end_points['prediction'] = prediction

    # training
    if output_data is not None:
        # [?,vocab_size]
        # labels=tf.one_hot(tf.reshape(output_data,[-1]),depth=vocab_size)
        # loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_logits,labels=labels)
        # total_loss = tf.reduce_mean(loss)
        # tf.summary.scalar('total_loss', total_loss)
        # end_points['loss'] = loss
        # end_points['total_loss'] = total_loss
        # Use sparse softmax
        labels = tf.reshape(output_data, [-1])
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_logits,labels=labels)
        total_loss = tf.reduce_mean(loss)
        with tf.name_scope("hidden"):
            tf.summary.histogram("embedding",embedding)
            tf.summary.histogram("output_weights",output_weights)
            tf.summary.histogram("output_bias", output_bias)
        tf.summary.scalar('total_loss', total_loss)
        end_points['loss'] = loss
        end_points['total_loss'] = total_loss

        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        end_points['train_op'] = train_op

    return end_points


class JinYongModel:
    def __init__(self):
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)

        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)

        if not os.path.exists(dict_path) or not os.path.exists(train_path):
            print('创建新的语料库')
            process_corpus(data_path)
        else:
            print('使用已有语料库')
        self.dict = _read_dict()
        for i,w in enumerate(self.dict):
            if w==SOS:
                self._sos_idx=i
            if w==EOS:
                self._eos_idx=i
        self._load_model()

    def _load_model(self):
        self.inputs=tf.placeholder(tf.int64,shape=[FLAGS.batch_size,None],name='input')
        self.targets = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, None],name='targets')

        self.end_points = rnn_model(FLAGS.cell_type, self.inputs, self.targets, vocab_size=len(self.dict), rnn_size=FLAGS.rnn_size
                               , num_layers=FLAGS.num_layers, learning_rate=FLAGS.learning_rate)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.inc_global_step_op = tf.assign_add(self.global_step, 1, name='inc_global_step')
        self.merge_summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(tf.global_variables())
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess=tf.Session()

        # run_options = tf.RunOptions()
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        self.train_writer = tf.summary.FileWriter(FLAGS.log_dir + "/train", self.sess.graph)
        self.sess.run(self.init_op, options=self.run_options)
        self.global_step_value = self.sess.run(self.global_step, options=self.run_options)

        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        if checkpoint:
            self.saver.restore(self.sess, checkpoint)
            print("从检查点[%s]加载模型，继续训练" % checkpoint, flush=True)
        else:
            print("找不到检查点[%s],从头开始训练" % FLAGS.model_dir, flush=True)

    def train(self):
        inputs,target= _read_train_data(FLAGS.batch_size)
        def get_feed_dict():
            i,t=self.sess.run([inputs,target])
            return {self.inputs:i,self.targets:i}

        try:
            while True:
                if self.global_step_value==0 or (self.global_step_value+1) % FLAGS.training_echo_interval == 0:
                    total_loss, _, _, _, self.global_step_value,summary = self.sess.run(
                        [self.end_points['total_loss'], self.end_points['last_state'], self.end_points['train_op'], self.inc_global_step_op,
                         self.global_step,self.merge_summary_op],feed_dict=get_feed_dict(),options=self.run_options)
                    self.train_writer.add_summary(summary, self.global_step_value)
                    self.train_writer.flush()
                    print('[%s]global step %d, Training Loss: %.10f' % (
                    time.strftime('%Y-%m-%d %H:%M:%S'),self.global_step_value,total_loss),flush=True)
                else:
                    _,_=self.sess.run([self.end_points['train_op'],self.inc_global_step_op],feed_dict=get_feed_dict(),options=self.run_options)

                if self.global_step_value%FLAGS.save_checkpoints_steps==0:
                    # 保存模型
                    self.saver.save(self.sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=self.global_step_value)
                    print('[%s]保存模型[global_step = %d]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.global_step_value),
                          flush=True)

                if self.global_step_value%FLAGS.gen_line_interval==0:
                    # 生成例句
                    sentence=self._gen_sentence(-1)
                    print('[%s]生成例句[global_step = %d]:%s' % (time.strftime('%Y-%m-%d %H:%M:%S'),self.global_step_value,sentence),
                          flush=True)
        except (StopIteration):
            self.saver.save(self.sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=self.global_step_value)
            print('[%s]训练结束, 保存模型[global_step = %d]' %(time.strftime('%Y-%m-%d %H:%M:%S'), self.global_step_value),flush=True)

    def _to_word_idx(self,predict):
        # pick word according to probability from predict
        predict /= np.sum(predict)
        sample = np.random.choice(np.arange(len(predict)), p=predict)
        return sample

    def _gen_sentence(self,begin_word_idx):

        if begin_word_idx >=0 and begin_word_idx<len(self.dict):
            word_idx=begin_word_idx
        else:
            # 随机生成首字
            # 构造batch_size大小的输入，但实际起用的只是第一行
            x =np.repeat(self._sos_idx,FLAGS.batch_size).reshape(FLAGS.batch_size,1)

            predict, last_state = self.sess.run([self.end_points['prediction'], self.end_points['last_state']],
                                                 feed_dict={self.inputs:x})
            word_idx=self._to_word_idx(predict[0])

        sentence=[]
        while word_idx!=self._eos_idx:
            if len(sentence)>80:
                break

            # 构造batch_size大小的输入，但实际起用的只是第一行
            x = np.repeat(word_idx, FLAGS.batch_size).reshape(FLAGS.batch_size, 1)
            predict, last_state = self.sess.run([self.end_points['prediction'], self.end_points['last_state']],
                                                 feed_dict={self.inputs: x})
            word_idx=self._to_word_idx(predict[0])
            if word_idx!=self._sos_idx:
                sentence.append(word_idx)

        # 将句子idx转换为文字
        sentence=''.join(list(map(self.dict.__getitem__,sentence)))
        return sentence

    def gen(self):
        while True:
            word = input('输入第一个汉字:')
            word_idx=-1
            for i,v in enumerate(self.dict):
                if word==v:
                    word_idx=i
                    break
            if (word_idx<0):
                print('输入不在字典内，随机生成数据:',end='')
            sentence=self._gen_sentence(word_idx)
            print("[%s]" % sentence, flush=True)
if __name__ == '__main__':
        print_args()

        model=JinYongModel()
        if FLAGS.mode == 'train':
            model.train()
        elif FLAGS.mode == 'gen':
            model.gen()