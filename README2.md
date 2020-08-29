# Learning patterns in strings with RNNs

Goal of this experiment is see how well a simple RNN can learn simple patterns in strings.

## Strings with simple patterns

A string pattern is defined as follows: [(1, 'a'), (2, 'b'), (9,'c'), (10,'d')]. A pattern with this string has an 'a' at index 1, a 'b' at index, a 'c' at index 9 and a 'd' at index 10. The other characters of the string are undefined by this pattern so they can be arbitrary characters. We manually define a list of patterns:

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

    nabnfzzxhcdbdnxogyll
    
    gabvvpfnycdllmmalfsq
    
    jggenbizyjkzevrkdpdy
    
    gxyrjvsrrvbqstxvqwig
    
    eabegbdsycdvscwlgmwl


