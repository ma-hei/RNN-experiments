# Learning patterns in strings with RNNs

Goal of this experiment is to see how well a simple RNN can learn simple patterns in strings. This experiment was created by roughly following the tutorial given here: https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html (with some modifications). 

## Strings with simple patterns

A string pattern is defined as follows: [(1, 'a'), (2, 'b'), (9,'c'), (10,'d')]. A pattern with this string has an 'a' at index 1, a 'b' at index 2, a 'c' at index 9 and a 'd' at index 10. The other characters of the string are undefined by this pattern so they can be arbitrary characters. We manually define a list of patterns:

    string_patterns = [
        [(1, 'a'), (2, 'b'), (9,'c'), (10,'d')],
        [(1, 'x'), (2, 'y'), (9,'v'), (10,'b')],
        [(1, 'e'), (2, 'f'), (9,'g'), (10,'h')],
        [(1, 'j'), (2, 'k'), (9,'l'), (10,'m')]
    ]

We generate n training strings of length l (l must be larger than the largest index in string_patterns), where each training string has one of the patterns in string_patterns. For each training string we chose the pattern of string_patterns randomly. Each training string ends with a new line.

    def replace_char_at_index(s, idx, c):
        return s[:idx] + c + s[idx + 1:]
    
    def get_string_with_pattern(string_pattern, string_length):
        rand_string = ''.join(random.choice(string.ascii_lowercase) for x in range(string_length))
        for (idx, char) in string_pattern:
            rand_string = replace_char_at_index(rand_string, idx, char)
        rand_string = rand_string + '\n'
        return rand_string
    
    def get_training_strings(n_strings, string_length):
        train_strings = []
        for i in range(n_strings):
            pattern = random.choice(string_patterns)
            train_string = get_string_with_pattern(pattern, string_length)
            train_strings.append(train_string)
        return train_strings

Example: 5 training strings of length 20.

    training_strings = get_training_strings(5, 20)
    for s in training_strings:
        print(s)

Gives the following:

    jjkjqmwdulmkaggacbzs
    
    bjkcgitwwlmfdzabubdm
    
    qefglqoerghfjcbxigxg
    
    sxyabgqaavbneftjchez
    
    ajkxbnhftlmmlkkpizky

With this we can generate batchified training data and a gluon Vocab as follows:

    def batchify_training_data(tokens, batch_size, vocab):
        batches = [tokens[i*(batch_size):(i+1)*batch_size] for i in range(int(len(tokens)/batch_size))]
        # temp_x is a list of all batches (100 training_strings, batch_size=32) -> 3 batches
        # Each item in temp_x is a ndarray of dimension len(training_string)xbatch_size (each column is a training_string)
        # we have batch_size columns per item in temp_x
        temp_x = [mxnet.nd.array([vocab(training_string[:-1]) for training_string in batch]).T for batch in batches]
        # temp_y is the corresponding next item for each training string in temp_train
        temp_y = [mxnet.nd.array([vocab(training_string[1:]) for training_string in batch]).T for batch in batches]
        training_data = list(zip(temp_x, temp_y))
        return training_data
    
    def generate_training_data(training_string_length, n_training_strings, batch_size):
        training_strings = get_training_strings(n_training_strings, training_string_length)
        tokens = [list(training_string) for training_string in training_strings]
        tokens_flattened = [item for sublist in tokens for item in sublist]
        counter = gluonnlp.data.count_tokens(tokens_flattened)
        vocab = gluonnlp.Vocab(counter)
        batchified_training_data = batchify_training_data(tokens, batch_size, vocab)
        return batchified_training_data, vocab

Example: Generate 100 training strings, each with length 20. Batchify the training strings into batches of 32 strings:

    training_data, vocab = generate_training_data(20, 100, 32)
    print(type(training_data)) 
    print(len(training_data))
    print(type(training_data[0]))
    print(type(training_data[0][0]))
    print(training_data[0][0].shape)
    print(training_data[0][1].shape)
    print(training_data[0][0][:,0])
    print(training_data[0][1][:,0])
    print(vocab)

Gives the following:

    <class 'list'>
    3
    <class 'tuple'>
    <class 'mxnet.ndarray.ndarray.NDArray'>
    (20, 32)
    (20, 32)
    
    [29. 18.  8. 12. 28. 18. 11. 28. 15. 14.  4. 25. 30. 22. 28. 11. 19. 22.
      4. 27.]
    <NDArray 20 @cpu(0)>
    
    [18.  8. 12. 28. 18. 11. 28. 15. 14.  4. 25. 30. 22. 28. 11. 19. 22.  4.
     27.  6.]
    <NDArray 20 @cpu(0)>
    Vocab(size=31, unk="<unk>", reserved="['<pad>', '<bos>', '<eos>']")

Explanation: We see that training_data is a list of length 3. Reason: 100 training strings in batches of 32 -> 3 batches. (last 4 training strings are discarded). Each batch is a tuple. The first element of each tuple is training data, the second element is labels. Training data and labels are NDArrays of shape (20,32) (each column represents a training string/label string, 32 strings per batch). Above, we take the first batch and print the first column of both the training and label matrix. Training strings and labels are now represented by indices in the vocab. Indices can be mapped back to characters by using vocab.idx_to_token. Comparing the training data and label for a particular training string (for a particular column) we can see that the label is the training string "offset" by one position. This makes sense because the goal of the RNN will be to predict the next character of a string, given a prefix. The vocabulary is of size 31: 26 alphabetic characters + new line, unk, pad, bos and eos. (Todo: why is new line not eos?)

## Building a RNN

    We can now build a RNN by subclassing gluons Block class
    
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

Training of the model looks as follows:

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
        print(exp(acc_l/n))

We see that we pass in a loss function and an updated function. Those are described below. To validate the model, we also need to predict characters given a prefix. We do this as follows:

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

Now we can generate some training data, create a model, train the model and after each iteration over the training data generate a string with this model. After each iteration over the training data, we generate a prefix that contains the first two characters of each pattern. We can then observe if the suffix generated by the model contains the last two characters in each pattern.

    def main():
        batch_size = 32
        string_length = 12
        n_training_strings = 1000
        batchified_training_data, vocab = generate_training_data(string_length, n_training_strings, batch_size)
        #temp = one_hot_encode(batchified_training_data[0][0], vocab)
        model = get_rnn(512, len(vocab))
        model.initialize(force_reinit=True, ctx = mxnet.cpu(), init=init.Normal(0.01))
        lr = 0.2
        trainer = mxnet.gluon.Trainer(model.collect_params(),'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
        loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
        #print(batchified_training_data[0][0].shape[1])
        for i in range(200):
            print("epoch: " + str(i))
            train_model(model, batchified_training_data, loss, updater, mxnet.cpu())
            sample_string_from_model_for_patterns(model, string_patterns, string_length, vocab) 

We can see that for a training string length of 12, after roughly 100 training epochs, the suffixes sampled from the model contain the patterns of the training strings:

    epoch: 53
    loss: 
    [7.6413903]
    <NDArray 1 @cpu(0)>
    cabbafabocdh 
    output has pattern
    bxymmnabvvbg
    output has pattern
    kefnjoyzvghy
    output has pattern
    tjkszsdghlmb
    output has pattern
