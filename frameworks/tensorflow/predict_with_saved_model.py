import tensorflow as tf
import numpy as np
from sklearn.datasets import load_svmlight_file


# prepare input data
train_datafile = "test_samples"
# prepare model
ckpt_file = "model.ckpt-199"

feature_num = 186 # 特征维度
data = load_svmlight_file(train_datafile, zero_based=True, n_features=feature_num)
features = data[0]
labels = data[1]
data_num = features.shape[0]  # 预测样本个数

features = features.toarray()
print features.shape
print labels.shape


with tf.Session() as sess:
    # load saved model from checkpoint file
    saver = tf.train.import_meta_graph(ckpt_file + ".meta")
    saver.restore(sess, ckpt_file)

    # check some tensor in graph
    print dir(sess)
    print sess._graph.get_tensor_by_name('inputs/positive:0')
    print sess._graph.get_tensor_by_name("embedding_weights:0")
    eb = sess.run('embedding_weights:0') # 输出一批参数看看
    #print eb

    #for op in tf.get_default_graph().get_operations():
    #    print str(op.name)

    #print tf.global_variables()


    # 首先要对网络内部结构非常清楚，才知道自己调用run时想要得到什么
    # predict using saved model
    final_pos = sess.run('RegLayer/Tanh:0',
                                feed_dict={'inputs/pos:0':features,
                                        'inputs/neg:0':np.random.rand(data_num, feature_num),
                                        'inputs/label_po:0': np.random.rand(data_num, 3),
                                        'inputs/label_neg:0': np.random.rand(data_num, 3),
                                        })

    print("reg_pos = {}").format(final_pos)

