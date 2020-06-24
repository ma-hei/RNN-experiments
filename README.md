# RNN experiments
## Learning patterns in strings

Training data: strings of length n where all strings share certain patterns.

Example: All strings begin with either 'a' or 'b'. A string that begins with 'a' always ends with 'c'. A string that begins with 'b' always ends with 'd'. All characters between the first and last character are randomly chosen from [A-Za-z]. The number of characters between the first and last character is fixed. With n=5, training strings can be akcWcvc, bcWWybd.

Goal: build a RNN with mxnet and gluon from scratch and observe if the RNN can learn simple patterns in the training strings. The RNN is built from scratch by following the tutorial given here https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html with some modifications.

### Creating a guonnlp vocabulary for the experiment:

A gluonnlp vocabulary is created as follows:

    temp = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
    counter = nlp.data.count_tokens(temp)
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None, bos_token=None, eos_token='\n')

    print(len(vocab))
    for i in range(len(vocab)):
        print(vocab.idx_to_token[i])

We can see that the vocabulary has 56 tokens (A-Z, a-z, "\n", unk, pad, bos).

### Generating training/test strings:
 
Training strings are generated with a fixed value for n_noise, where n_noise determines the number of characters between the first and last character. The first and last character of all train and test strings follow a certain pattern.

    def get_string_with_first_last_pattern(first_char, last_char, n_between):
        temp_string = first_char
        for i in range(n_between):
            temp_string = temp_string + random.choice(string.ascii_letters)
        temp_string = temp_string + last_char + '\n'
        return temp_string

    n_noise = 20

    for i in range(5):
        print(get_string_with_first_last_pattern('a', 'b', n_noise))

We can see that strings generated this way begin with 'a' and end on 'b'

    aqIeXsfOYVEmzghzWIECOb
    aAfmnXyYaoKJccfHwjkuTb
    aAPqOIpzcVYdcFVVOHDNfb
    aJmnGErMdhEHNTKIOiTmSb
    aNSWYgqbdSbpejiKRwsCyb

### Building a RNN from scratch

Following tutorial: https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html with some modifications.

#### The RNN parameters

First, we define the RNN parameters:
 
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

In each time step, one character of the training string is fed into the RNN and each character is one-hot-encoded. Since we have 56 tokens in the vocabulary, the number of input units is 56 (see next section). num_hiddens can be adjusted arbitrarily. ctx is used to tell mxnet if we're using a gpu or cpu. I'm using a cpu locally.

#### One-hot-encoding of characters

To feed training strings into the RNN, we need a one-hot-encoding of training strings. We will use gluonnlp's batchify class for this. Here's an example of how this works:

The RNN is trained on sequences of fixed length. In this case the sequence length is 2 + n_noise (1 start-character + n_noise noise-characters + 1 end-character).

    sequence_length = n_noise + 2
    batch_size = 1

    batchify = nlp.data.batchify.CorpusBPTTBatchify(vocab, sequence_length, 1, last_batch='keep')
    single_train_string = get_string_with_first_last_pattern('a', 'b', n_noise)
    rnn_train_data = batchify(single_train_string)

    print(type(rnn_train_data))
    print(len(rnn_train_data))
    print(rnn_train_data[0])

We see that rnn_train_data is a SimpleDataset of length 1. Printing the single entry shows a tuple where the first item represents the training string from the first character to the last character and the second item represents the string from the second character until the new line character. Thus, in each training step, the second item will be the learning target.

Next, the "forward-pass" of the RNN:

    def rnn(inputs, state, params):
        # Inputs shape: (num_steps, 1, vocab_size)
        W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state
        outputs = []
        for X in inputs:
            H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
            Y = np.dot(H, W_hq) + b_q
            outputs.append(Y)
        return np.concatenate(outputs, axis=0), (H,)

In the for loop, we're iterating over each character of the training string.
Getting the initial parameters of the RNN and initializing the state:

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

Now the RNN class:

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

We can now create an instance of the RNN and feed a training string into the (untrained) RNN.

    num_hiddens, ctx = 512, mx.cpu()

    model = RNNModelScratch(len(vocab), num_hiddens, ctx, get_params,
                        init_rnn_state, rnn)

    state = model.begin_state(batch_size=1, ctx=ctx)
    result = model(rnn_train_data[0], state)

    print(type(result))
    print(len(result))
    print(result[0].shape)
    print(type(result[1]))
    print(len(result[1]))
    print(result[1][0].shape)

We can see that we get back a tuple of length 2. The first element is the the RNN's output at each character-step. The second element is the hidden state of the RNN after the last character was fed into the RNN.

Before training the model we need two helper methods. One is grad_clipping, which scales the parameters in order to avoid the parameters of the model to "explode" through backpropagation. The other one is gd (gradient descent) which is used to update the parameter models. We're also defining a loss function that we will use to train the model. We're using softmax cross entropy loss here.

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

Now the actual training of the model

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

The train method gets a number of train/target (X/Y) sequences as input, as well as a loss function (softmax cross entropy loss) and an updater (gd). Now we just need to call this method with appropriate training data. Each training episode we're either generating a training string that begins with an 'a' and ends with a 'b' or we generate one that begins with a 'c' and ends on a 'd'. The target string is always the same as the training string shifted by one character to the right (train="axyzb" -> target ="xyzb\n"). The training strings have the patterns discussed above.

    def train_ch8_changing_train_string(model, vocab, sequence_length, loss, lr, num_epochs, ctx):
    
        def updater(model):
            return gd(model)
    
        for epoch in range(num_epochs):
    
            string1 = ""
            if (epoch%2 == 0):
                string1 = string1 + get_string_with_first_last_pattern('a', 'b', sequence_length-2)
            else:
                string1 = string1 + get_string_with_first_last_pattern('c', 'd', sequence_length-2)
    
            bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(vocab, sequence_length + 1, batch_size, last_batch='keep')
            train_data = bptt_batchify(string1)
    
            l = train_epoch_ch8(model, train_data, loss, updater, ctx)
            if epoch % 100 == 0:
                print("done with epoch " + str(epoch))
                print("loss: " + str(l))
                print(predict_ch8('a', sequence_length, model, vocab, ctx))
                print(predict_ch8('c', sequence_length, model, vocab, ctx))
    
Every 100 episodes, we're sampling two sequences from the RNN: one sequence that begins with an 'a' and one sequence that begins with a 'c'. This will give us some idea of how well the RNN has learned the pattern in the training strings up to this episode. predict_ch8 is defined as follows:

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

We can observe that after a few thousand epochs, the RNN has learned that strings beginning with 'a' are ending on 'b' and strings beginning with 'c' are ending with 'd'. The RNN has also learned that the ending 'b'/'d' is followed by a new line character:

    done with epoch 9100                                                                                                                                                                                 loss: 2.852933
    aCCgZeb

    cCCgZed                                                                                                                                                                                              
    done with epoch 9200                                                                                                                                                                                 loss: 2.8556256
    aCDCZKb

    cbDbZbd                                                                                                                                                                                              
    done with epoch 9300                                                                                                                                                                                 loss: 2.880085
    affYUVb

    cfYUfdd

