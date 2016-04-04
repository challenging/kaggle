import sys
import pickle
import pprint

with open(sys.argv[1], "rb") as INPUT:
    o = pickle.load(INPUT)
    pprint.pprint(o)
