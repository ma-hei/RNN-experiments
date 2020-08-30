import random
import string
import gluonnlp
import mxnet

# define a list of string patterns. A pattern is 
# a list of tuples. The first element of a tuple 
# is an index and the second element is the character
# of the string with this pattern at the index.
# Example for a string with pattern [ (1,'a'), (5,'d')]:
#    acyfduhgn
# Notice that the first character is an 'a' and the 5th
# character is a 'd'.
string_patterns = [
    [(1, 'a'), (2, 'b'), (9,'c'), (10,'d')],
    [(1, 'x'), (2, 'y'), (9,'v'), (10,'b')],
    [(1, 'e'), (2, 'f'), (9,'g'), (10,'h')],
    [(1, 'j'), (2, 'k'), (9,'l'), (10,'m')]
]

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

def main():

    training_strings = get_training_strings(5, 20)
    for s in training_strings:
        print(s)  

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

if __name__ == '__main__':
    main()
