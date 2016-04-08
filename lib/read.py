import sys
import pickle
import operator

o = None
with open(sys.argv[1], "rb") as INPUT:
    o = pickle.load(INPUT)

print "done"
for i in o:
    print i
