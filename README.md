# RNN experiments
## Learning patterns in strings

Training data: strings of length n where all strings share certain patterns.

Example: All strings begin with either 'a' or 'b'. A string that begins with 'a' always ends with 'c'. A string that begins with 'b' always ends with 'd'. All characters between the first and last character are randomly chosen from [A-Za-z]. The number of characters between the first and last character is fixed. With n=5, training strings can be akcWcvc, bcWWybd.

### Creating a guonnlp vocabulary for the experiment:

A gluonnlp vocabulary is created as follows:

    temp = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
    counter = nlp.data.count_tokens(temp)
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None, bos_token=None, eos_token='\n')

    print(len(vocab))
    for i in range(len(vocab)):
        print(vocab.idx_to_token[i])

We can see that the vocabulary has 53 tokens (A-Z, a-z, "\n").

### Generating training/test strings:
 
Training strings are generated with a fixed value for n_noise, where n_noise determines the number of characters between the first and last character. The first and last character of all train and test strings follow a certain pattern.

    def get_string_with_first_last_pattern(first_char, last_char, n_between):
        temp_string = first_char
        for i in range(n_between):
            temp_string = temp_string + random.choice(string.ascii_letters)
        temp_string = temp_string + last_char + '\n'
        return temp_string

    for i in range(5):
        print(get_string_with_first_last_pattern('a', 'b', 20))
