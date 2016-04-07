import sys
import pickle
import operator

c1, c2 = 0, 0
with open(sys.argv[1], "rb") as INPUT:
    o = pickle.load(INPUT)

    for key, value in o.items():
        if isinstance(value, float):
            print key, value
        else:
            for sub_key, ratio in value.items():
                print key, sub_key, ratio
