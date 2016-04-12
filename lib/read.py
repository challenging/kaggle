import sys
import pickle
import operator

o = None
with open(sys.argv[1], "rb") as INPUT:
    o = pickle.load(INPUT)

for i, v in o.items():
    print i, v
