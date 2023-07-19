def add_num_sign(f):
    def dec_f(*args, **kwargs):
        res_ls = f(*args, **kwargs)
        return ['#' + r for r in res_ls]
    return dec_f



@add_num_sign
def qualitative():
    return ['ff1f5b', '009ade', '00cd6c', 'af58ba', 'ffc61a', 'f28522', 'a0b1ba', 'a6761d']

@add_num_sign
def cold_qualitative():
    return ['7b7b7c', '28a8de', 'fff300', 'f3835e', 'ef5a29', 'f1eee8']

@add_num_sign
def blue_sequential():
    return ['e4f1f7', 'c5e1ef', '9ec9e2', '6cb0d6', '3c93c2', '226e9c', '0d4a70']

@add_num_sign
def green_sequential():
    return ['e1f2e3', 'cde5d2', '9ccea7', '6cba7d', '40ad5a', '228b3b', '06592a']

@add_num_sign
def pink_sequential():
    return ['f9d8e6', 'f2acca', 'ed85b0', 'e95694', 'e32977', 'c40f5b', '8f003b']

def qualitative_gen():
    i = 0
    l = len(qualitative())
    while True:
        yield qualitative()[i % l]