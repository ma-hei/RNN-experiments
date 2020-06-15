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

We see that rnn_train_data is a SimpleDataset of length 1. Printing the single entry shows a tuple where the first item represents the training string from the first character to the new line symbol and the second item represents the string from the second character until the new line character. In each training step, the second item will be the learning target.

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

The inputs parameter will always be one training string in which each character is represented by its one-hot-encoding.  
