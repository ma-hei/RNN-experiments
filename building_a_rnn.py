from d2l import Accumulator
from mxnet import nd, init, autograd
from mxnet.gluon import rnn, nn
from mxnet.ndarray import exp
from generating_training_data import *
import math

class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)
        
    def forward(self, inputs, state):
        X = nd.one_hot(inputs, self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

def get_rnn(n_hidden, vocab_size):
    rnn_layer = rnn.RNN(512)
    rnn_layer.initialize()
    model = RNNModel(rnn_layer, vocab_size)
    return model

def train_model(model, train_data, loss, updater, ctx):
    state = None
    batch_size = train_data[0][0].shape[1]
    metric = Accumulator(2)
    acc_l, n = 0, 0
    for X, Y in train_data:
        state = model.begin_state(batch_size)
        y = Y.reshape(-1)
        with autograd.record():
            py, state = model(X, state)
            l = loss(py, y).mean()
        l.backward()
        grad_clipping(model, 1)
        updater(batch_size=1)
        acc_l += l * y.size
        n += y.size
    print("loss: " + str(exp(acc_l/n)))

def grad_clipping(model, theta):
    if isinstance(model, mxnet.gluon.Block):
        params = [p.data() for p in model.collect_params().values()]
    else:
        params = model.params
    norm = mxnet.ndarray.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def predict(prefix, num_predicts, model, vocab, device):
    state = model.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: mxnet.nd.array([outputs[-1]], ctx=device).reshape(1, 1)
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_predicts):
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(axis=1).reshape(1).asnumpy()))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# generates a string with the pattern and returns first half of the string
def generate_prefix_for_pattern(model, string_pattern, string_length):
    string_with_pattern = get_string_with_pattern(string_pattern, string_length)
    return string_with_pattern[0:int(string_length/2)]
     
# this method assumes that all patterns in string_patterns have at least 4 elements and
# that the third and fourth pattern element are in the second half of the string
def sample_string_from_model_for_patterns(model, string_patterns, string_length, vocab):
    for p in string_patterns:
        string_with_prefix = generate_prefix_for_pattern(model, p, string_length)
        output = predict(string_with_prefix, string_length-len(string_with_prefix), model, vocab, mxnet.cpu())
        print(output)
        if output[p[2][0]] == p[2][1] and output[p[3][0]] == p[3][1]:
            print("output has pattern")

def main():
    batch_size = 32
    string_length = 12
    n_training_strings = 1000
    batchified_training_data, vocab = generate_training_data(string_length, n_training_strings, batch_size)
    model = get_rnn(512, len(vocab))
    model.initialize(force_reinit=True, ctx = mxnet.cpu(), init=init.Normal(0.01))
    lr = 0.2
    trainer = mxnet.gluon.Trainer(model.collect_params(),'sgd', {'learning_rate': lr})
    updater = lambda batch_size: trainer.step(batch_size)
    loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(200):
        print("epoch: " + str(i))
        train_model(model, batchified_training_data, loss, updater, mxnet.cpu())
        sample_string_from_model_for_patterns(model, string_patterns, string_length, vocab) 

if __name__ == '__main__':
    main()
