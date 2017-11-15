from random import shuffle
import numpy as np
# Read file
with open('hw1_15_train.dat', 'r') as f:
    lines = f.readlines()
data = []
for line in lines:
	s = line.split( )
	s.insert(0, 1.0)
	data.append(s);
	
def sdsign(v):
	if v > 0:
		return 1.0
	else:
		return -1.0
def check(d, w):
	x = np.array(d[0:5], dtype=float)
	y = float(d[5])
	v = np.dot(x, w)
	return sdsign(v)

w = np.zeros(5) # = threshold
count = 0
again = 1
while again:
	for d in data:
		if check(d, w) != float(d[5]):
			count += 1
			w = w + float(d[5]) * np.array(d[0:5], dtype=float)
	again = 0
	for d in data:
		if check(d, w) != float(d[5]):
			again = 1
			break
print count

