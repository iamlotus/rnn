import time
import os
import nltk
import itertools
import numpy as np


unknown_token = "UNKNOWN_TOKEN"
file_start_token = "FILE_START"
file_end_token = "FILE_END"





def file_data_generator(src_root):
    for root,dirs,files in os.walk(src_root):
        for python_file in filter(lambda file:file.endswith('.py'),files):
            url=os.path.join(root,python_file)
            with open(url,'r',buffering=8192,encoding='utf-8') as file:
                yield file.read()

def get_data(src_root,vocabulary_size):
    start=time.time()
    #Tokenize the sentences into words
    files = [nltk.sent_tokenize(data.lower()) for data in file_data_generator(src_root)]
    files= ['%s %s %s' % (file_start_token, f, file_end_token) for f in files]
    tokenized_files=[nltk.word_tokenize(sent) for sent in files]
    # Count word frequencies
    word_freq=nltk.FreqDist(itertools.chain(*tokenized_files))
    print ("Found %d unique word tokens"%(len(word_freq.items())))

    #Get the most common words and build index_to_word and word_to_index vectors
    vocab=word_freq.most_common(vocabulary_size-1)
    index_to_word=[x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index=dict([(w,i) for i,w in enumerate(index_to_word)])

    print("Using vocabulary size %d"%vocabulary_size)
    print("The least frequent word in vocabulary is '%s' with %d times"%(vocab[-1][0],vocab[-1][1]))

    # Replace all words not in vocabulary with unkown_token
    for i,sent in enumerate(tokenized_files):
        tokenized_files[i]= [w if w in word_to_index else unknown_token for w in sent]

    print('read %d python files from "%s" in %d seconds' % (len(tokenized_files), src_root, time.time() - start))
    print(tokenized_files)

    # Create training data
    X_train=np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_files])
    y_train=np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_files])

    return X_train,y_train