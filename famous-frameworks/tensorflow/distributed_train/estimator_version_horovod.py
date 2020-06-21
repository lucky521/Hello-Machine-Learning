#coding=utf-8
import horovod.tensorflow as hvd
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.INFO)

hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Build a dummy dataset
d = 16
n_examples = 100
X = np.vstack([
    np.random.normal(10, 2, (n_examples, d)),
    np.random.normal(-10, 2, (n_examples, d))
])
y = np.hstack([
    np.repeat(0, n_examples),
    np.repeat(1, n_examples)
])
print('X', X.shape)
print('y', y.shape)

# Define input fn
def input_fn(features, labels, batch_size, shuffle, repeat):
    if labels is not None:
        inputs = (features, labels)
    else:
        inputs = features
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(1000)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    # For distributed training, needs to return dataset
    return dataset

    # return dataset.make_one_shot_iterator().get_next()

train_input_fn = lambda: input_fn(X, y, batch_size=128, shuffle=True, repeat=True)
test_input_fn = lambda: input_fn(X, y, batch_size=128, shuffle=False, repeat=False)
predict_input_fn = lambda: input_fn(X, None, batch_size=128, shuffle=False, repeat=False)

def model_fn(features, labels, mode, params):
    """
    features: `x` from input_fn
    labels: `y` from input_fn
    mode: either TRAIN, EVAL, or PREDICT
    params: hyperparams e.g. learning rate
    """

    # Define model
    net = features
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    probs = tf.nn.softmax(logits)
    preds = tf.argmax(probs, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'probs': probs,
                'preds': preds
            })

    else:
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

        opt = hvd.DistributedOptimizer(opt)

        train = opt.minimize(loss=loss, global_step=tf.train.get_global_step())

        metrics = {'accuracy': tf.metrics.accuracy(labels, preds)}

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train,
            eval_metric_ops=metrics
        )

    return spec

params = {
    'learning_rate': 1e-4,
    'hidden_units': [16],
    'n_classes': 2
}

model_dir = './test_model_r' + str(hvd.rank()) 

model = tf.estimator.Estimator(
    model_fn=model_fn,
    params=params,
    model_dir=model_dir)

bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

model.train(input_fn=train_input_fn, steps=2000, hooks=[bcast_hook])

result = model.evaluate(input_fn=test_input_fn)
print("result:", result)

pred = model.predict(input_fn=predict_input_fn)
for p in pred:
    print("pred:", p)
