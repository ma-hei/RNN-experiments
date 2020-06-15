import gluonnlp as nlp
import mxnet as mx
import random
import string

temp = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
counter = nlp.data.count_tokens(temp)
vocab = nlp.Vocab(counter, eos_token='\n')
for i in range(len(vocab)):
    print(vocab.idx_to_token[i])

n_noise = 20

def get_string_with_first_last_pattern(first_char, last_char, n_noise):
    temp_string = first_char
    for i in range(n_noise):
        temp_string = temp_string + random.choice(string.ascii_letters)
    temp_string = temp_string + last_char + '\n'
    return temp_string

for i in range(5):
    print(get_string_with_first_last_pattern('a', 'b', n_noise), end="")

sequence_length = n_noise + 2
batch_size = 1
batchify = nlp.data.batchify.CorpusBPTTBatchify(vocab, sequence_length, 1, last_batch='keep')
single_train_string = get_string_with_first_last_pattern('a', 'b', n_noise)
rnn_train_data = batchify(single_train_string)

print(type(rnn_train_data))
print(len(rnn_train_data))
print(rnn_train_data[0])


