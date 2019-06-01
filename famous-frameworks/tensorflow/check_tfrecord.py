import tensorflow as tf

target_file = "/Users/liulu51/jd-git/jd-deep-model/cornerstone/cpj/test/test_data_dnn_pointwise/jiadian_train_2018-12-08_2018-12-08_tf_record_000"
target_file = "/Users/liulu51/jd-git/jd-deep-model/cornerstone/cpj/test/test_data_jingpai/train_1343_deep_ranking_part_0"

example_cnt = 0
for example in tf.python_io.tf_record_iterator(target_file):
    result = tf.train.Example.FromString(example)
    print("result", result)
    example_cnt += 1


print("example_cnt = ", example_cnt)
