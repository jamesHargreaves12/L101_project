import csv
import os

caches = {}
cache_files = {}


def _make_safe(val):
    return str(val).replace(",", "_-comma-_").replace("\n", "_-newline-_")


def _undo_safe(val):
    return val.replace("_-comma-_", ",").replace("_-newline-_", "\n")


def cache_call(func, args):
    f_name = func.__name__
    if str(args) in caches[f_name]:
        return caches[f_name][str(args)]
    else:
        value = func(args)
        if "is_in_database" not in f_name and "drqa" not in f_name:
            print("Uncached call({}): {} result: {}".format(f_name, args, str(value)[:10]))
        caches[f_name][str(args)] = value
        cache_files[f_name].write(
            "{},{}\n".format(_make_safe(args).replace("\"", "_-dq-_"), _make_safe(value)))
        return value


def setup_cache(func, cache_filename=None):
    f_name = func.__name__
    if not cache_filename:
        cache_filename = f_name
    assert (f_name not in caches)
    file_path = "caches/{}.txt".format(cache_filename)

    if not os.path.exists(file_path):
        cache_files[f_name] = open(file_path, "w+")
        caches[f_name] = {}
    else:
        cache_files[f_name] = open(file_path, "a+")
        caches[f_name] = {}
        cache_files[f_name].seek(0)
        for line in csv.reader(cache_files[f_name]):
            caches[f_name][_undo_safe(line[0]).replace("_-dq-_","\"")] = _undo_safe(line[1])


if __name__ == "__main__":
    def test_function(x):
        print("     CALLED", x)
        return x + x


    def print_test(val, pred):
        if val != pred:
            print(val == pred, val, pred)
        else:
            print(True)


    setup_cache(test_function)

    print_test(cache_call(test_function, "str1"), "str1str1")
    print_test(cache_call(test_function, ""), "")
    test_str = "long long long, string with annoying chars\n"
    print_test(cache_call(test_function, test_str), test_str * 2)

    print_test(cache_call(test_function, "str1"), "str1str1")
    print_test(cache_call(test_function, ""), "")
    print_test(cache_call(test_function, test_str), test_str * 2)
