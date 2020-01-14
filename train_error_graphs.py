from math import log, exp

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def get_data(path):

    fp = open(path)

    ys = []
    xs = []
    for line in fp.readlines():
        vals = line.split(",")
        ys.append((float(vals[1])))
        xs.append(int(vals[0]))
    return xs,ys

no_sent = get_data("data/mlp_training_error/mlp-final_no_sent_ents.mdl")
with_sent = get_data("data/mlp_training_error/mlp-final_with_sent_ents.mdl")

plt.plot(no_sent[0][675:],no_sent[1][675:])
# plt.plot(with_sent[0], with_sent[1])
plt.show()
