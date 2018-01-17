import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
def getdata(filename):
	data = np.loadtxt(filename, dtype='f')
	N, dim = data.shape
	X = np.zeros((N, dim))
	X.fill(1)
	X[:,1:] = data[:, 0:-1]
	Y = np.zeros((N,1))
	Y.fill(1)
	Y[:,0] = data[:, -1]
	return X, Y

def split(X, Y):
	X_train = X[0:120,:]
	X_test = X[120:200,:]
	Y_train = Y[0:120,:]
	Y_test = Y[120:200,:]
	return X_train, Y_train, X_test, Y_test

def sign(y):
	if y >= 0:
		return 1
	return -1

def error(X, Y, W, N):
	cnt = 0
	H = np.dot(X, W)
	for i in range(0, N):
		if sign(H[i]) != Y[i][0]:
			cnt += 1
	return float(cnt)/N

def regularization(X, Y, N, lamb):
	inverse = inv( np.dot(X.transpose(), X) + lamb*np.identity(dim))
	return np.dot(inverse, np.dot(X.transpose(), Y))

X, Y = getdata("hw4_train.dat")
N, dim = X.shape
X_train, Y_train, X_val, Y_val = split(X, Y)
N_train = 120
N_val = 80
X_test, Y_test = getdata("hw4_test.dat")
N_test, dim = X_test.shape

etrain = [None]*13
evalu = [None]*13
eout = [None]*13
for log in range(-10, 3):
	lamb = 10**log
	w_reg = regularization(X_train, Y_train, N_train, lamb)
	Etrain = error(X_train, Y_train, w_reg, N_train)
	Eval = error(X_val, Y_val, w_reg, N_val)
	Eout = error(X_test, Y_test, w_reg, N_test)
	etrain[log+10] = Etrain
	evalu[log+10] = Eval
	eout[log+10] = Eout
	print "lambda_log:", log, " Etrain:", Etrain, " Eval:", Eval, " Eout:", Eout


t = list(range(-10, 3))
plt.xlabel('log_lambda')
plt.plot(t, etrain, 'b', t, evalu, 'r')
plt.legend(['Etrain', 'Eval'])
plt.show()

