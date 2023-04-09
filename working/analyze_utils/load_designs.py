from design import Design
from film import FilmSimple
from spectrum import SpectrumSimple
import numpy as np
import os
import re
import pickle
# Load trained films data and construct Design objects


def load_designs(file_dir, filter: list[str]=[]) -> list[Design]:
    fname_ls = os.listdir(file_dir)
    designs = []
    for fname in fname_ls:
        if not all([s in fname for s in filter]):
            continue
        with open(file_dir + fname, 'rb') as f:
            designs.append(pickle.load(f))
    return designs