#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import fastText
from tensorflow.contrib.tensorboard.plugins import projector
from pprint import pprint

LOGDIR = "tensorflow_logs/do-embedding_v1"
data_bin = 'data/wiki.zh.bin'

# load model
'''
# need fasttext from gensim
word2vec = fasttext.load_model(data_bin)
pprint(vars(word2vec))
'''


# need fastTest from Facebook github
f = fastText.load_model(data_bin)
words = f.get_words()
print(str(len(words)) + " " + str(f.get_dimension()))
'''
for w in words:
    v = f.get_word_vector(w)
    vstr = ""
    for vi in v:
        vstr += " " + str(vi)
    try:
        #print(w + vstr)
        pass
    except IOError as e:
        if e.errno == errno.EPIPE:
            pass
'''

# create a list of vectors
embedding = np.empty((len(words), f.get_dimension()), dtype=np.float32)
for i, word in enumerate(words):
    embedding[i] = f.get_word_vector(word)





# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# write labels
with open(os.path.join(LOGDIR, 'metadata.tsv'), 'w') as f:
    label_count = 0
    for word in words:
        f.write(word.encode('utf-8') + '\n')
        label_count += 1
print("label count={}").format(label_count)

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(summary_writer, config)

# save the model
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))


print("Check: tensorboard --logdir=./tensorflow_logs")

