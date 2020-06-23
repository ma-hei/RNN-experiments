import gluonnlp as nlp
import math
import mxnet as mx
from mxnet import autograd, np, npx
import random
import string
npx.set_np()

def get_string_with_first_last_pattern(first_char, last_char, n_noise):
    temp_string = first_char
    for i in range(n_noise):
        temp_string = temp_string + random.choice(string.ascii_letters)
    temp_string = temp_string + last_char + '\n'
    return temp_string


def rnn(inputs, state, params):
    # Inputs shape: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    count = 0
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)

def get_params(vocab_size, num_hiddens, ctx):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=ctx)
    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = np.zeros(num_hiddens, ctx=ctx)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=ctx)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

def init_rnn_state(batch_size, num_hiddens, ctx):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )

class RNNModelScratch:
    """A RNN Model based on scratch implementations."""

    def __init__(self, vocab_size, num_hiddens, ctx,
                 get_params, init_state, forward):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, ctx)
        self.init_state, self.forward_fn = init_state, forward

    def __call__(self, X, state):
        X = npx.one_hot(X, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)

    def collect_params(self):
        return self.params

def grad_clipping(model, theta):
    params = model.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def gd(params):
    for param in params:
        param[:] = param - 0.1 * param.grad

loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

def predict_ch8(prefix, num_predicts, model, vocab, ctx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    outputs = [vocab[prefix[0]]]

    def get_input():
        return np.array([outputs[-1]], ctx=ctx).reshape(1, 1)
    for y in prefix[1:]:  # Warmup state with prefix
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_predicts):  # Predict num_predicts steps
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def train_epoch_ch8(model, train_iter, loss, updater, ctx):
    loss_total = 0
    for X, Y in train_iter:

        X = X.as_np_ndarray()
        Y = Y.as_np_ndarray()

        state = model.begin_state(batch_size=1, ctx=ctx)

        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(ctx), y.as_in_ctx(ctx)  

        with autograd.record():
            py, state = model(X, state)
            l = loss(py, y).mean()

        loss_total += l
        l.backward()
        grad_clipping(model, 1)

        updater(model.params)

    return loss_total/len(train_iter)

def train_ch8_changing_train_string(model, vocab, sequence_length, loss, lr, num_epochs, ctx):

    def updater(model):
        return gd(model)

    for epoch in range(num_epochs):

        string1 = ""
        if (epoch%2 == 0):
            string1 = string1 + get_string_with_first_last_pattern('a', 'b', sequence_length-2)
        else:
            string1 = string1 + get_string_with_first_last_pattern('c', 'd', sequence_length-2)

        bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(vocab, sequence_length, batch_size, last_batch='keep')
        train_data = bptt_batchify(string1)

        l = train_epoch_ch8(model, train_data, loss, updater, ctx)
        if epoch % 100 == 0:
            print("done with epoch " + str(epoch))
            print("loss: " + str(l))
            print(predict_ch8('a', sequence_length, model, vocab, ctx))
            print(predict_ch8('c', sequence_length, model, vocab, ctx))

temp = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
counter = nlp.data.count_tokens(temp)
vocab = nlp.Vocab(counter, eos_token='\n')
for i in range(len(vocab)):
    print(vocab.idx_to_token[i])

n_noise = 5

for i in range(5):
    print(get_string_with_first_last_pattern('a', 'b', n_noise), end="")

sequence_length = n_noise + 2
batch_size = 1
batchify = nlp.data.batchify.CorpusBPTTBatchify(vocab, sequence_length, 1, last_batch='keep')
single_train_string = get_string_with_first_last_pattern('a', 'b', n_noise)
rnn_train_data = batchify(single_train_string)

num_hiddens, ctx = 512, mx.cpu()

model = RNNModelScratch(len(vocab), num_hiddens, ctx, get_params,
                        init_rnn_state, rnn)

state = model.begin_state(batch_size=1, ctx=ctx)
result = model(rnn_train_data[0][0].as_np_ndarray(), state)

print(type(result))
print(len(result))
print(result[0].shape)
print(type(result[1]))
print(len(result[1]))
print(result[1][0].shape)

print(type(rnn_train_data))
print(len(rnn_train_data))

loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
train_ch8_changing_train_string(model, vocab, sequence_length, loss, 1, 10000, ctx)

