# -*- coding: utf-8 -*-

import re
import numpy as np
import math
import tensorflow as tf
import os
import time
import operator


tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_integer('rnn_size', 128, 'rnn hidden size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'rnn num layers.')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_string('data_type', 'poems', 'jinyong/poems')
tf.app.flags.DEFINE_string('train_corpus_name', 'train_corpus.txt', 'train corpus name')
tf.app.flags.DEFINE_string('cell_type', 'rnn', 'rnn/gru/lstm')
tf.app.flags.DEFINE_integer('epochs', 500, 'train how many epochs.')
tf.app.flags.DEFINE_integer('training_echo_interval', 10, 'echo train logs interval.')
tf.app.flags.DEFINE_integer('training_save_interval', 200, 'save model interval during training.')
tf.app.flags.DEFINE_string('mode','train' , 'train/gen, train model or gen poems use model')
tf.app.flags.DEFINE_string('cuda_visible_devices', '0', '''[Train] visible GPU ''')

FLAGS=tf.app.flags.FLAGS

SOS="SOS"
EOS="EOS"


def read_dict():
    """
       读出词典 word_idx_map，附加SOS和EOS
       :return:
    """

    dict_path = os.path.join('data', FLAGS.data_type, 'dict.txt')
    with open(dict_path, 'r', encoding='utf') as f:
        word_idx_map = {}
        for line in f.read().split('\n')[:-1]:
            m=re.match(r'(\d+):(.)',line)

            word_idx_map[m.group(2)] = int(m.group(1))

    # 增加SOS和EOS
    if SOS not in word_idx_map:
        sos_id = max(word_idx_map.values()) + 1
        word_idx_map[SOS]=sos_id

    if EOS not in word_idx_map:
        eos_id = max(word_idx_map.values()) + 1
        word_idx_map[EOS] = eos_id

    print('【加载词典】%s， 共%d词'%(dict_path,len(word_idx_map)), flush=True)
    return word_idx_map

def read_corpus():
    """
    读出所有训练语料，前后加上SOS和EOS
    :return:
    """
    word_idx_map=read_dict()
    sos_id=word_idx_map[SOS]
    eos_id=word_idx_map[EOS]

    train_path = os.path.join('data', FLAGS.data_type, FLAGS.train_corpus_name)
    with open(train_path, 'r', encoding='utf') as f:
        train_vectors=[]
        for line in f.read().split('\n')[:-1]:
            try:
                line_vector=[sos_id]+list(map(word_idx_map.__getitem__,line))+[eos_id]
            except KeyError as e:
                print(e)
            train_vectors.append(line_vector)
    print('【加载训练集】%s， 共%d行' % (train_path, len(train_vectors)), flush=True)
    return word_idx_map,train_vectors


def batch_generator(batch_size, line_vectors, fill_value):
    """
    预先生成所有padding后的结果，避免动态padding，这里line_vector应该已经在头尾分别填充了SOS和EOS
    最后一个batch 有可能不满，全部填充为fill_value
    :param batch_size:
    :param line_vectors:
    :param fill_value:
    :return:
    """

    batch_num=math.ceil(len(line_vectors) / batch_size)
    x_batches=[]
    y_batches=[]
    for i in range(batch_num):
        batch_start = i * batch_size
        batch_end = min((i+1) * batch_size, len(line_vectors))
        batch_data = line_vectors[batch_start:batch_end]
        batch_times= max(len(line)-1 for line in batch_data)
        # padding with fill_value
        x_data= np.full((batch_size,batch_times),fill_value=fill_value,dtype=np.int32)
        y_data= np.full((batch_size,batch_times),fill_value=fill_value,dtype=np.int32)
        for row,data in enumerate(batch_data):
            x_data[row, :len(data)-1]=data[:-1]
            y_data[row, :len(data)-1] = data[1:]
        """
        x_data      y_data
        [START,A,B,FILL_VALUE]  [A,B,END,FILL_VALUE]
        [START,C,D,E]  [C,D,E,END]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)

    while True:
        for i in range(batch_num):
            yield x_batches[i],y_batches[i]



def rnn_model(model,input_data,output_data,vocab_size,rnn_size,num_layers,learning_rate):
    """
    Create a RNN model 
    :param model: cell model, one of 'rnn'/'gru'/'lstm'
    :param input_data: 
    :param output_data: output_data tensor of shape (batch,time) is used as labels to calculate loss, None in predictions
    :param vocab_size: vocabulary size,
    :param rnn_size: the depth of internal neural network
    :param num_layers:
    :param learning_rate:
    :return: endpoints
    """

    argkws={'num_units':rnn_size}

    if model=='rnn':
        cell_fun=tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun=tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun =tf.nn.rnn_cell.LSTMCell
        argkws['state_is_tuple']=True
    else:
        raise ValueError('wrong model "%s"'%model)

    end_points={}

    batch_size= input_data.get_shape()[0]
    cell=cell_fun(**argkws)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state=cell.zero_state(batch_size,tf.float32) if output_data is not None else cell.zero_state(1,tf.float32)
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
    end_points['last_state'] = last_state

    # training
    if output_data is not None:
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

    # prediction
    else:
        prediction=tf.nn.softmax(output_logits)
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points


def run_training():

    model_dir=os.path.join("model",FLAGS.data_type)
    log_dir=os.path.join("logs",FLAGS.data_type)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("【创建模型目录】%s"%model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print("【创建日志目录】%s"%log_dir)


    word_idx_map, train_vectors= read_corpus()
    fill_value = word_idx_map[' ']
    train_bg=batch_generator(FLAGS.batch_size, train_vectors, fill_value)

    steps_per_epoch=math.ceil(len(train_vectors)/FLAGS.batch_size)
    max_global_step=FLAGS.epochs*steps_per_epoch

    input_data=tf.placeholder(tf.int32,[FLAGS.batch_size,None])
    output_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    end_points=rnn_model(FLAGS.cell_type,input_data,output_data,vocab_size=len(word_idx_map),rnn_size=FLAGS.rnn_size
                         ,num_layers=FLAGS.num_layers,learning_rate=FLAGS.learning_rate)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inc_global_step_op = tf.assign_add(global_step, 1, name='inc_global_step')
    merge_summary_op=tf.summary.merge_all()


    # saver
    saver=tf.train.Saver(tf.global_variables())
    init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        run_options = tf.RunOptions()
        # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        train_writer=tf.summary.FileWriter(os.path.join(log_dir,"train"),sess.graph)
        sess.run(init_op,options=run_options)
        global_step_value=sess.run(global_step,options=run_options)

        checkpoint=tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            saver.restore(sess,checkpoint)
            print("加载已有模型[%s], 继续训练"%checkpoint,flush=True)
        else:
            print("找不到模型[%s], 重新训练"%model_dir,flush=True)

        try:
            while True:
                if global_step_value >= max_global_step:
                    print('[%s] Train completed with %d epochs, %d steps' % (
                    time.strftime('%Y-%m-%d %H:%M:%S'), FLAGS.epochs, global_step_value),flush=True)
                    return

                input_data_value,output_data_value=next(train_bg)
                if global_step_value==0 or (global_step_value+1) % FLAGS.training_echo_interval == 0:
                    total_loss, _, _, _, global_step_value,summary = sess.run(
                        [end_points['total_loss'], end_points['last_state'], end_points['train_op'], inc_global_step_op,
                         global_step,merge_summary_op],
                        feed_dict={input_data: input_data_value, output_data: output_data_value},options=run_options)
                    epoch = math.ceil(global_step_value / steps_per_epoch) -1
                    batch = global_step_value - (epoch * steps_per_epoch)
                    train_writer.add_summary(summary, global_step_value)
                    train_writer.flush()
                    print('[%s] Epoch %d, Batch %d, global step %d, Training Loss: %.8f' % (
                    time.strftime('%Y-%m-%d %H:%M:%S'),epoch, batch,global_step_value,total_loss),flush=True)

                else:
                    _,_,global_step_value=sess.run([end_points['train_op'],inc_global_step_op,global_step],
                        feed_dict={input_data: input_data_value, output_data: output_data_value},options=run_options)

                if global_step_value % FLAGS.training_save_interval ==0 or global_step_value>=max_global_step:
                    # save every epoch
                    saver.save(sess, os.path.join(model_dir,'model'), global_step=global_step_value)
                    print('[%s] Save model[global_step = %d]' % (time.strftime('%Y-%m-%d %H:%M:%S'),global_step_value),flush=True)

        except KeyboardInterrupt:
            print('Interrupt manually, try saving checkpoint for now')
            saver.save(sess, os.path.join(model_dir,'model'), global_step=global_step_value)
            print('Save model[global_step = %d]' % global_step_value,flush=True)


class SentenceGen:

    @staticmethod
    def to_word(predict, vocabs):
        # pick word according to probability from predict
        predict = predict[0]
        predict /= np.sum(predict)
        sample = np.random.choice(np.arange(len(predict)), p=predict)
        if sample > len(vocabs):
            return vocabs[-1]
        else:
            return vocabs[sample]

    def __init__(self):
        model_dir = os.path.join("model", FLAGS.data_type)
        self._word_idx_map = read_dict()
        self._words= list(map(lambda x:x[0],sorted(self._word_idx_map.items(),key=operator.itemgetter(1))))

        self._batch_size=1
        self._input_data = tf.placeholder(tf.int32, [self._batch_size, None], name='input_data')
        self._end_points = rnn_model(FLAGS.cell_type, input_data=self._input_data, output_data=None, vocab_size=len(self._word_idx_map), rnn_size=FLAGS.rnn_size,
                               num_layers=FLAGS.num_layers,learning_rate=FLAGS.learning_rate)

        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(model_dir)
        self._sess=tf.Session()
        self._sess.as_default()
        run_options = tf.RunOptions()
        # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        self._sess.run(init_op,options=run_options)

        if checkpoint:
            saver.restore(self._sess, checkpoint)
            print("加载模型[%s]" % checkpoint, flush=True)
        else:
            raise ValueError('can not find model ')

    def gen(self,begin_word):
        if not begin_word in self._word_idx_map:
            print('汉字不在词汇表中，随机生成诗句',end=' ',flush=True)
            begin_word=None

        x = np.array([[self._word_idx_map[SOS]]])
        predict, last_state = self._sess.run([self._end_points['prediction'], self._end_points['last_state']],
                                             feed_dict={self._input_data: x})

        if begin_word:
            word=begin_word
        else:
            word=SentenceGen.to_word(predict, self._words)


        i = 0
        poem_=''
        while word != EOS:
            poem_ += word
            i += 1
            if i >= 100:
                break
            x = np.zeros((1, 1))
            x[0, 0] = self._word_idx_map[word]
            predict, last_state = self._sess.run([self._end_points['prediction'], self._end_points['last_state']],
                                           feed_dict={self._input_data: x, self._end_points['initial_state']: last_state})
            word = SentenceGen.to_word(predict, self._words)
        return poem_

def run_gen():
    poemGen=SentenceGen()
    while True:
        word = input('输入第一个汉字:')
        poem=poemGen.gen(word)
        print("[%s]"%poem,flush=True)


def main(_):

    print("系统参数:")
    for param in ['batch_size','rnn_size','num_layers','learning_rate','data_type','train_corpus_name','cell_type','epochs'
        ,'training_echo_interval','training_save_interval','mode','cuda_visible_devices']:
        print("%s = %s"%(param,FLAGS[param]._value),flush=True)

    if FLAGS.mode=='train':
        run_training()
    elif FLAGS.mode=='gen':
        run_gen()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices  # set GPU visibility in multiple-GPU environment
    tf.app.run()

