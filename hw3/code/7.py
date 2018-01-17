import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
N = 1000
T = 1000

def generate_X():
	X = np.zeros((N,3))
	x1x2 = np.random.random_sample((N, 2))*2-1
	X[:,1:] = x1x2
	return X

def calculate_Y(X):
	Y = np.zeros((N,1))
	for i in range(0, N):
		Y[i][0] = np.sign(X[i][1]**2+X[i][2]**2-0.6)

	flip = random.sample(range(N), N/10)
	for f in flip:
		Y[f][0] *= -1
	return Y

def transform(X):
	Z = np.zeros((N,6))
	for i in range(0, N):
		Z[i][0] = 1
		Z[i][1] = X[i][1]
		Z[i][2] = X[i][2]
		Z[i][3] = X[i][1]*X[i][2]
		Z[i][4] = X[i][1]**2
		Z[i][5] = X[i][2]**2
	return Z

def weight(X, Y):
	W = np.zeros((6,1))
	W = np.dot( np.dot(inv(np.dot(X.transpose(),X)), X.transpose()), Y)
	return W

def error(Z, W, Y):
	H = np.dot(Z, W)
	cnt = 0
	for i in range(0, N):
		if np.sign(H[i][0]) != Y[i][0]:
			cnt += 1
	return float(cnt)/N

Ein = [None]*T
Eout = [None]*T
avg_Ein = 0.0
avg_Eout = 0.0

for i in range(0, T):
	X_train = generate_X()
	Y_train = calculate_Y(X_train)
	Z_train = transform(X_train)

	X_test = generate_X()
	Y_test = calculate_Y(X_test)
	Z_test = transform(X_test)

	W = weight(Z_train, Y_train)

	Ein[i] = error(Z_train, W, Y_train)
	Eout[i] = error(Z_test, W, Y_test)
	# print "Ein =", Ein[i], "Eout =", Eout[i]
	avg_Ein += Ein[i]
	avg_Eout += Eout[i]

print "Avg.Ein =", avg_Ein/T
print "Avg.Eout =", avg_Eout/T

# Draw histogram
bin_num = 12
arr = plt.hist(Eout, bins=bin_num, facecolor='b', edgecolor='black')
plt.xlabel('Eout')
plt.ylabel('Frequency')
for i in range(bin_num):
    plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))
plt.show()