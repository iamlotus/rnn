import tensorflow as tf
import os
import collections

tf.app.flags.DEFINE_string('data_type', 'poems', 'data type.')
tf.app.flags.DEFINE_string('corpus_name', 'corpus.txt', 'the corpus name')
tf.app.flags.DEFINE_string('train_corpus_name', 'train_corpus.txt', 'the processed corpus name')
tf.app.flags.DEFINE_string('dict_name', 'dict.txt', 'the dict name')
tf.app.flags.DEFINE_integer('min_content_length', 3, 'minimal length of content that accepted as train')
tf.app.flags.DEFINE_integer('max_content_length', 4000, 'maximum length of content that accepted as train')

FLAGS=tf.app.flags.FLAGS

def _poem_line_filter_fn(line):
    if ':' not in line:
        return None,True

    title, content = line.strip().split(':')
    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
        return None, True

    return content,False

def _jinyong_line_filter_fn(line):
    return line,False

def process_corpus():
    """
    预处理预料，删除过短过长及不合理的行
    :return:
    """
    if FLAGS.data_type=='jinyong':
        data_dir="data/jinyong"
        line_filter_fn = _jinyong_line_filter_fn
    elif FLAGS.data_type=='poems':
        data_dir = "data/poems"
        line_filter_fn = _poem_line_filter_fn
    else:
        raise ValueError('unkown data_type %s'%FLAGS.data_type)

    corpus_path=os.path.join(data_dir,FLAGS.corpus_name)
    print("处理数据集%s" % (corpus_path))
    with open(corpus_path, 'r', encoding='utf') as f:
        error_line = 0
        total_line = 0
        lines = []
        for line in f.readlines():
            # remove \r \n
            line=line[:-1].strip()
            total_line += 1

            try:
                content,error = line_filter_fn(line)
                if error:
                    error_line+1
                    continue

                if len(content) < FLAGS.min_content_length or len(content) > FLAGS.max_content_length:
                    error_line += 1
                    continue

                lines.append(content)
            except ValueError as e:
                error_line += 1

    all_words = [word for line in lines for word in line]
    # 确保有BLANK,后续padding会用到
    all_words.append(' ')
    counter = collections.Counter(all_words)
    counter_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    words, _ = zip(*counter_pairs)
    word_idx_map = dict(zip(words, range(len(words))))
    # train_vectors = [list(map(lambda word: word_idx_map[word], line)) for line in lines]

    print("【数据集】共%d行，%d行无法处理，剩余%d行。【共计】%d词"%(total_line,error_line,len(lines),len(word_idx_map)))

    dict_path=os.path.join(data_dir,FLAGS.dict_name)
    with open(dict_path, 'w', encoding='utf') as f:
        for e in word_idx_map:
            f.write('%d:%s\n' % (word_idx_map[e], e))
    print("【生成词典文件】%s"%dict_path)

    train_corpus_name=os.path.join(data_dir,FLAGS.train_corpus_name)
    with open(train_corpus_name, 'w', encoding='utf') as f:
        for line in lines:
            f.write('%s\n'%line)
    print("【生成训练预料库(文本)】%s" % train_corpus_name)

    # train_data_name=os.path.join(data_dir,FLAGS.train_data_name)
    # with open(train_data_name, 'w', encoding='utf') as f:
    #     for vec in train_vectors:
    #         f.write('%s\n'%vec)
    # print("【生成训练预料库(向量)】%s" % train_data_name)



if __name__=='__main__':
    process_corpus()

