import pickle


def save(fname, *objs):
    dict = {}
    for i, o in enumerate(objs):
        dict[i] = o

    with open(fname, 'wb') as f:
        pickle.dump(dict, f)


def load(fname):
    with open(fname, 'rb') as f:
        res = pickle.load(f)

    ret_objs = []
    for key in res:
        ret_objs.append(res[key])

    return ret_objs
