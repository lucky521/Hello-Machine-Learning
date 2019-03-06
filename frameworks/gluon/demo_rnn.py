# python3.6
import d2lzh as d2l
import math, time
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
print("Size of corpus", len(corpus_indices)) # 以idx代替char来表示整个语料
print("Size of vocab", vocab_size) #预料中一共有多少种char


def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)


def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)

    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hq, b_q]

    for param in params:
        param.attach_grad()
    return params

def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


params = get_params()



def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, 
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        (Y, state) = rnn(X, state, params)
        #print('Y', Y)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])

predict_output = predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens,
                    vocab_size, ctx, idx_to_char, char_to_idx)
print('predict_output', predict_output)



def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                for s in state:
                    s.detach()
            with autograd.record():
                #print('X.shape', X.shape)
                #print('Y.shape', Y.shape)
                inputs = to_onehot(X, vocab_size)
                (outputs, state) = rnn(inputs, state, params)
                print('inputs', inputs)
                print('outputs', outputs)
                outputs = nd.concat(*outputs, dim=0)
                y = Y.T.reshape((-1,))
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)
            d2l.sgd(params, lr, 1)
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' %
                (epoch + 1, math.exp(l_sum/n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char,char_to_idx))


num_epochs = 250
num_steps = 35
batch_size = 32
lr = 1e2
clipping_theta = 1e-2
pred_period = 50
pred_len = 50
prefixes = ['分开', '不分开']

train_and_predict_rnn(rnn, get_params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, corpus_indices,
                    idx_to_char, char_to_idx, True, num_epochs, num_steps,
                    lr, clipping_theta, batch_size, pred_period,
                    pred_len, prefixes)
