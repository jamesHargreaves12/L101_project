import csv

mins, maxs = None, None


def get_col_min_max(data):
    mins = data[0].copy()
    maxs = data[0].copy()
    for row in data:
        for i, x in enumerate(row):
            mins[i] = min(mins[i], x)
            maxs[i] = max(maxs[i], x)
    return mins, maxs


def normalise_rows(data):
    global mins
    global maxs
    if not mins:
        with open("data/mins_and_maxs_for_normalisation.txt", "r") as fp:
            mins = [float(x) for x in fp.readline().split(",")]
            maxs = [float(x) for x in fp.readline().split(",")]
    output = []
    for row in data:
        output.append([(x - mins[i]) / (maxs[i] - mins[i]) for i, x in enumerate(row)])
    return output


def get_features_from_file(path):
    Xs = []
    ys = []
    count = 0
    for line in csv.reader(open(path, "r")):
        X = [float(x) for x in line[:-1]]
        y = int(line[-1])
        Xs.append(X)
        ys.append(y)
        count += 1
    return Xs, ys


def recalc_mins_maxs():
    Xs, _ = get_features_from_file("data/nn_train_with_drqa_dev.csv")
    Xs.extend(get_features_from_file("data/nn_train_with_drqa_full.csv")[0])
    mins, maxs = get_col_min_max(Xs)
    with open("data/mins_and_maxs_for_normalisation.txt", "w") as fp:
        fp.write(",".join([str(x) for x in mins])+"\n")
        fp.write(",".join([str(x) for x in maxs])+"\n")


if __name__ == "__main__":
    recalc_mins_maxs()
