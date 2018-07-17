import os
import tensorflow as tf
import numpy as np
import fastText
from tensorflow.contrib.tensorboard.plugins import projector

# load model
word2vec = fastText.load_model('wiki.en.bin')

LOGDIR = "tensorflow_logs/do-embedding_v1"

# create a list of vectors
embedding = np.empty((len(word2vec.words), word2vec.dim), dtype=np.float32)
for i, word in enumerate(word2vec.words):
    embedding[i] = word2vec[word]

# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# write labels
with open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w') as f:
    for word in word2vec.words:
        f.write(word + '\n')

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join(LOGDIR, 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

# save the model
saver = tf.train.Saver()
saver.save(sess, os.path.join('LOGDIR', "model.ckpt"))