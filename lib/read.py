import sys
import pickle
import operator

c1, c2 = 0, 0
with open(sys.argv[1], "rb") as INPUT:
    o = pickle.load(INPUT)

    for layer1, info in o.items():
        for layer2, value in sorted(info.items(), key=operator.itemgetter(0)):
            if value > float(sys.argv[2]):
                print layer1, layer2, value
                c1 += 1
            else:
                c2 += 1

print c1, c2
