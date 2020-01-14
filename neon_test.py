from random import random
from time import time

from neon.backends import gen_backend

from baseline.normalization import normalise_rows

be = gen_backend(batch_size=1)

import csv
import numpy as np
from neon.callbacks.callbacks import Callbacks
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, Misclassification
from neon import logger as neon_logger

from neon.data.dataiterator import ArrayIterator


class MyModel(object):
    def __init__(self, model_name, batch_size=128):
        global be
        be = gen_backend(batch_size=batch_size)
        self.model_name = model_name
        self.init_norm = Gaussian(loc=0.0, scale=0.01)

        # setup model layers
        self.layers = [Affine(nout=100, init=self.init_norm, activation=Rectlin()),
                       Affine(nout=2, init=self.init_norm, activation=Logistic(shortcut=True))]

        # initialize model object
        self.model = Model(layers=self.layers)

    def load_from_path(self, load_model_path):
        print("loading model from:", load_model_path)
        self.model.load_params(load_model_path)

    def save_model(self, save_model_path):
        print("Saving model at:", save_model_path)
        self.model.save_params(save_model_path)

    def get_results(self, dataset: list):
        """

        :param dataset: list of list of features (if you want to run a single example run [example]
        :return:
        """
        input = ArrayIterator(np.array(dataset), name='input')
        return self.model.get_outputs(input)

    def train(self, train: tuple, valid: tuple, epoch=10, dev_set_result_path=None):
        train_X = np.array(train[0])
        train_y = np.array(train[1])
        test_X = np.array(valid[0])
        test_y = np.array(valid[1])
        dev_set_result_path = dev_set_result_path or "data/mlp_training_error/" + self.model_name

        train_set = ArrayIterator(train_X, train_y, make_onehot=True, nclass=2, name='train')
        valid_set = ArrayIterator(test_X, test_y, make_onehot=True, nclass=2, name='valid')

        # setup cost function as CrossEntropy
        cost = GeneralizedCost(costfunc=CrossEntropyBinary())

        # setup optimizer
        optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

        self.model.initialize(train_set, cost)

        # configure callbacks
        callbacks = Callbacks(self.model, eval_set=valid_set)

        error_rate = self.model.eval(valid_set, metric=Misclassification())
        neon_logger.display('Before Misclassification error = %.3f%%' % (error_rate * 100))

        # run fit
        self.model.fit(train_set, optimizer=optimizer,
                       num_epochs=epoch, cost=cost, callbacks=callbacks, valid_set=valid_set,
                       valid_res_path=dev_set_result_path)

        error_rate = self.model.eval(valid_set, metric=Misclassification())
        neon_logger.display('Misclassification error = %.3f%%' % (error_rate * 100))


# def normalise(data):
#     maximums = [0 for _ in data[0]]
#     for d in data:
#         for i, e in enumerate(d):
#             maximums[i] = max(maximums[i], e)
#     results = []
#     for d in data:
#         results.append([x / maximums[i] for i, x in enumerate(d)])
#     return results


def get_data_from_file(path):
    Xs = []
    ys = []
    count = 0
    for line in csv.reader(open(path, "r")):
        X = [float(x) for x in line[:-1]]
        y = int(line[-1])
        Xs.append(X)
        ys.append(y)
        count += 1
    return normalise_rows(Xs), ys


def run_model(train_path, test_path, from_model=False, model_name=None, resave_model=False, epoch=10):
    if from_model:
        assert model_name, "Must supply model name to train from model"
    elif not model_name:
        model_name = "mlp-{}.mdl".format(time())

    print("Using model:", model_name)
    model_path = "models/test/" + model_name

    train_data = get_data_from_file("" + train_path)
    test_data = get_data_from_file("" + test_path)

    model = MyModel(model_name)
    if from_model:
        model.load_from_path(model_path)
    model.train(train_data, test_data, epoch=epoch)
    if not from_model or resave_model:
        model.save_model(model_path)


if __name__ == "__main__":
    # this method is not meant to be general do not use it without understanding it
    def _balence_classes(datavalues):
        inputs,outputs = datavalues
        percent_1s = sum(outputs)/len(outputs)
        print(percent_1s)
        result_Xs, result_ys = [],[]
        for X, y in zip(inputs, outputs):
            if y == 1 or random() < percent_1s:
                result_Xs.append(X)
                result_ys.append(y)
        return result_Xs, result_ys


    train_path = "data/nn_train_with_drqa_full_3.csv"
    test_path = "data/nn_train_with_drqa_valid_no_sent_ent_2.csv"
    model_name = "mlp-final_no_sent_ents.mdl"

    # train_path = "data/nn_train_with_drqa_full_with_capital_ents_2.csv"
    # test_path = "data/nn_train_with_drqa_valid_with_sent_ent_2.csv"
    # model_name = "mlp-final_with_sent_ents.mdl"

    train_path = "data/nn_train_with_drqa_full_3.csv"
    test_path = "data/nn_train_with_drqa_valid_no_sent_ent_2.csv"
    model_name = "mlp-final_no_sent_no_drqa.mdl"
    drqa_sent_index = 8


    train_data = get_data_from_file(train_path)
    test_data = get_data_from_file(test_path)
    print("LEN: ", len(train_data[0]), len(test_data[0]))

    train_data = _balence_classes(train_data)
    test_data = _balence_classes(test_data)
    print("LEN: ", len(train_data[0]), len(test_data[0]))


    print(train_data[0][0])
    train_data = [x[:8] + x[9:] for x in train_data[0]],train_data[1]
    test_data = [x[:8] + x[9:] for x in test_data[0]],test_data[1]
    print(train_data[0][0])


    model = MyModel(model_name, batch_size=256)
    model_path = "models/test/" + model_name
    # model.load_from_path(model_path)
    model.train(train_data, test_data, epoch=10000)
    model.save_model(model_path)


    #
    #
    # data = [[0.5, 19543, 1990, 2, 6, 1, 11612, 173.93871615712675, 150.54620305601156, 0, 0],
    #             [0.3333333333333333, 20894, 2029, 4, 6, 1, 125259, 11.739170653025699, 12.15710464652003, 0, 0],
    #             [0.5, 79142, 8115, 2, 8, 2, 1056669, 295.66546875775055, 177.34941335398952, 0, 0]]
    # true = [0, 0, 1]
    #
    # print(model.get_results(data))
