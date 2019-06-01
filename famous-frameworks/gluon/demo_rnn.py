# python3.6
import d2lzh as d2l
import math, time
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
print("Size of corpus", len(corpus_indices)) # 以idx代替char来表示整个语料
print("Size of vocab", vocab_size) #预料中一共有多少种char


# X的每行是一条样本，每列是一个样本维度
# 该函数为X的所有行分别构建为size维度的onehot向量，然后连成一个list
def to_onehot(X, size): 
    #print('Do onehot for {} of size {}'.format(X.shape, size))
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

# 生成全0的state
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )

# 区别state和params, state是中间状态， params是模型层参数
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    #print(H.shape) # 中间状态的维度
    #print(len(inputs), len(inputs[0]), len(inputs[0][0]),len(inputs[0][0][0])) # 样本被切分用于多次训练
    outputs = []
    for X in inputs:
        #print(X.shape) #每次通过网络的样本个数
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


# 初始一份state，拿已有的params
# X是prefix，去计算Y
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


#params = get_params()
#predict_output = predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens,
#                    vocab_size, ctx, idx_to_char, char_to_idx)
#print('predict_output', predict_output)


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
            #X:一个batch data是batch_size个长度为num_steps的char list，其中char由idx表示
            #Y: Y和X格式一样
            #print(X[0],Y[0]) 
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                for s in state:
                    s.detach() # Returns a new NDArray, detached from the current graph.
            with autograd.record():
                #print('X.shape', X.shape)
                #print('Y.shape', Y.shape)
                inputs = to_onehot(X, vocab_size)
                (outputs, state) = rnn(inputs, state, params)
                #print('inputs.shape', len(inputs), inputs[0].shape)
                #print('outputs.shape', len(outputs), outputs[0].shape)
                outputs = nd.concat(*outputs, dim=0) # 把outputs全连起来
                print('concat_outputs', outputs.shape)
                y = Y.T.reshape((-1,))
                print('y', y.shape)
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)
            d2l.sgd(params, lr, 1) # 真正的去优化params参数
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' %
                (epoch + 1, math.exp(l_sum/n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char,char_to_idx))


num_epochs = 250
num_steps = 35
batch_size = 32  #每次通过网络的样本个数
lr = 1e2  #学习率
clipping_theta = 1e-2
pred_period = 50
pred_len = 50
prefixes = ['分开', '不分开']
train_and_predict_rnn(rnn, get_params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, corpus_indices,
                    idx_to_char, char_to_idx, True, num_epochs, num_steps,
                    lr, clipping_theta, batch_size, pred_period,
                    pred_len, prefixes)
