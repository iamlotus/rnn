# -*- coding: utf-8 -*-

import collections
import numpy as np
import math
import tensorflow as tf
import os
import time

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_integer('rnn_size', 128, 'rnn hidden size.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('./model'), 'model save path.')
tf.app.flags.DEFINE_string('log_dir', os.path.abspath('./logs'), 'logs save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poems.txt'), 'file name of poems.')
tf.app.flags.DEFINE_string('cell_type', 'rnn', 'rnn/gru/lstm')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 500, 'train how many epochs.')
tf.app.flags.DEFINE_integer('training_echo_interval', 20, 'echo logs interval during training.')
tf.app.flags.DEFINE_integer('training_save_interval', 100, 'save model interval during training.')
tf.app.flags.DEFINE_string('mode','train' , 'train/gen, train model or gen poem use model')
FLAGS=tf.app.flags.FLAGS
start_token='B'
end_token='E'


def process_poems(file_path):
    """
    Process txt file specified by `file_path`
    :param file_path: a txt file, echo poem in one line
    :return: (poems_vector,word_idx_map,words), where poems_vector is list<list<wordIdx>>, word_idx_map is map<chinese word, wordIdx>, words is list of all words
    """
    # 诗集
    poems=[]
    error_line=0
    total_line=0
    with open(file_path,'r',encoding='utf') as f:
        for line in f.readlines():
            total_line+=1

            if not ':' in line:
                error_line+=1
                continue
            try:
                title , content=line.strip().split(':')
                content=content.replace(' ','')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content :
                    error_line+=1
                    continue

                content= '%s%s%s'%(start_token,content,end_token)
                poems.append(content)
            except ValueError as e:
                error_line += 1
                pass

    print("%d poems processed totally, %d poems error and skipped, %d poems left" %(total_line,error_line,len(poems)))

    all_words= [word for poem in poems for word in poem]
    # print("%d words totally"%len(all_words))

    counter=collections.Counter(all_words)
    counter_pairs= sorted(counter.items(),key=lambda x:x[1],reverse=True)
    words, freqs = zip(*counter_pairs)
    # padding with blank
    words = words +(' ',)
    word_idx_map=dict(zip(words,range(len(words))))
    poems_vector=[list(map(lambda word:word_idx_map[word],poem)) for poem in poems]
    return poems_vector,word_idx_map,words


def batch_generator(batch_size,poem_vec,fill_value):
    batch_num=math.ceil(len(poem_vec)/batch_size)
    x_batches=[]
    y_batches=[]

    for i in range(batch_num):
        batch_start = i * batch_size
        batch_end = min((i+1) * batch_size,len(poem_vec))
        batch = poem_vec[batch_start:batch_end]
        batch_times= max(len(poem)-1 for poem in batch)
        # padding with blank
        x_data= np.full((batch_size,batch_times),fill_value=fill_value,dtype=np.int32)
        y_data= np.full((batch_size,batch_times),fill_value=fill_value,dtype=np.int32)
        for row,data in enumerate(batch):
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



def rnn_model(model,input_data,output_data,vocab_size,rnn_size,learning_rate):
    """
    Create a RNN model 
    :param model: cell model, one of 'rnn'/'gru'/'lstm'
    :param input_data: 
    :param output_data: output_data tensor of shape (batch,time) is used as labels to calculate loss, None in predictions
    :param vocab_size: vocabulary size,
    :param rnn_size: the depth of internal neural network
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

    # training
    if output_data is not None:
        # [?,vocab_size]
        labels=tf.one_hot(tf.reshape(output_data,[-1]),depth=vocab_size)
        loss=tf.nn.softmax_cross_entropy_with_logits(output_logits,labels)
        total_loss=tf.reduce_mean(loss)
        tf.summary.scalar('total_loss', total_loss)
        train_op=tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state']=initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    # prediction
    else:
        prediction=tf.nn.softmax(output_logits)
        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points

def run_training():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    poems_vector, word_idx_map, words=process_poems(FLAGS.file_path)
    # idx_word_map = dict(map(lambda item: (item[1], item[0]), word_idx_map.items()))
    fill_value = word_idx_map[' ']
    bg=batch_generator(FLAGS.batch_size, poems_vector, fill_value)

    batch_num_per_epoch=math.ceil(len(poems_vector)/FLAGS.batch_size)
    max_global_step=FLAGS.epochs*batch_num_per_epoch

    input_data=tf.placeholder(tf.int32,[FLAGS.batch_size,None])
    output_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    end_points=rnn_model(FLAGS.cell_type,input_data,output_data,vocab_size=len(words),rnn_size=FLAGS.rnn_size,learning_rate=FLAGS.learning_rate)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inc_global_step_op = tf.assign_add(global_step, 1, name='inc_global_step')
    merge_summary_op=tf.summary.merge_all()


    # saver
    saver=tf.train.Saver(tf.global_variables())
    init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        writer=tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
        sess.run(init_op)
        global_step_value=sess.run(global_step)

        checkpoint=tf.train.latest_checkpoint(FLAGS.model_dir)
        if checkpoint:
            saver.restore(sess,checkpoint)
            print("Restore from checkpoint %s"%checkpoint)
        print("Start training")

        try:
            while True:
                if global_step_value >= max_global_step:
                    print('[%s] Train completed with %d epochs, %d steps' % (
                    time.strftime('%Y-%m-%d %H:%M:%S'), FLAGS.epochs, global_step_value))
                    return

                input_data_value,output_data_value=next(bg)
                if (global_step_value+1) % FLAGS.training_echo_interval == 0:
                    loss, _, _, _, global_step_value,summary = sess.run(
                        [end_points['total_loss'], end_points['last_state'], end_points['train_op'], inc_global_step_op,
                         global_step,merge_summary_op],
                        feed_dict={input_data: input_data_value, output_data: output_data_value})
                    epoch = math.ceil(global_step_value / batch_num_per_epoch)
                    batch = global_step_value - (epoch-1) * batch_num_per_epoch
                    print('[%s] Epoch %d, Batch %d, global step %d, Training Loss: %.8f' % (time.strftime('%Y-%m-%d %H:%M:%S'),epoch, batch,global_step_value,loss))
                    writer.add_summary(summary, global_step_value)
                    writer.flush()
                else:
                    _,_,global_step_value=sess.run([end_points['train_op'],inc_global_step_op,global_step],
                        feed_dict={input_data: input_data_value, output_data: output_data_value})
                    epoch = math.ceil(global_step_value / batch_num_per_epoch)
                    batch = global_step_value - (epoch - 1) * batch_num_per_epoch

                if global_step_value % FLAGS.training_save_interval ==0 or global_step_value>=max_global_step:
                    # save every epoch
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=global_step_value)
                    print('[%s] Save model[global_step = %d]' % (time.strftime('%Y-%m-%d %H:%M:%S'),global_step_value))

        except KeyboardInterrupt:
            print('Interrupt manually, try saving checkpoint for now')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=global_step_value)
            print('Save model[global_step = %d]' % (global_step_value))




class PoemGen:

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
        print('Loading corpus...')
        _, self._word_idx_map, self._words = process_poems(FLAGS.file_path)

        print('Loading model...')
        self._batch_size=1
        self._input_data = tf.placeholder(tf.int32, [self._batch_size, None], name='input_data')
        self._end_points = rnn_model(FLAGS.cell_type, input_data=self._input_data, output_data=None, vocab_size=len(self._words), rnn_size=FLAGS.rnn_size,
                               learning_rate=FLAGS.learning_rate)

        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        self._sess=tf.Session()
        self._sess.as_default()
        self._sess.run(init_op)

        if checkpoint:
            saver.restore(self._sess, checkpoint)
            print("Model %s loaded" % checkpoint)
        else:
            raise ValueError('can not find model ')

    def gen(self,begin_word):
        if not begin_word in self._word_idx_map:
            print('汉字不在词汇表中，随机生成诗句',end=' ')
            begin_word=None

        x = np.array([list(map(self._word_idx_map.get, start_token))])
        predict, last_state = self._sess.run([self._end_points['prediction'], self._end_points['last_state']],
                                             feed_dict={self._input_data: x})

        if begin_word:
            word=begin_word
        else:
            word=PoemGen.to_word(predict,self._words)


        i = 0
        poem_=''
        while word != end_token:
            poem_ += word
            i += 1
            if i >= 100:
                break
            x = np.zeros((1, 1))
            x[0, 0] = self._word_idx_map[word]
            predict, last_state = self._sess.run([self._end_points['prediction'], self._end_points['last_state']],
                                           feed_dict={self._input_data: x, self._end_points['initial_state']: last_state})
            word = PoemGen.to_word(predict, self._words)
        return poem_

def run_gen():
    poemGen=PoemGen()
    while True:
        word = input('输入第一个汉字:')
        poem=poemGen.gen(word)
        print("[%s]"%poem)


def main(_):
    print("FLAGS=%s"%FLAGS.__flags)

    if FLAGS.mode=='train':
        run_training()
    elif FLAGS.mode=='gen':
        run_gen()


if __name__ == '__main__':
    tf.app.run()

