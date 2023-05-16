import pickle
import dill



def save(fname, *objs):
    dict = {}
    for i, o in enumerate(objs):
        dict[i] = o

    with open(fname, 'wb') as f:
        dill.dump(dict, f)


def load(fname):
    try:
        with open(fname, 'rb') as f:
            res = dill.load(f)
    except Exception as e:
        print('Trying to load as pickled obj')
        with open(fname, 'rb') as f:
            res = pickle.load(f)
    ret_objs = []
    for key in res:
        ret_objs.append(res[key])

    return ret_objs
