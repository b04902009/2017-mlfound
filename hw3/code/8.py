import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
lr = 0.001
T = 2000
# T = 3

def getdata(filename):
	data = np.loadtxt(filename, dtype='f')
	N, dim = data.shape
	X = np.zeros((N, dim))
	X[:,1:] = data[:, 0:-1]
	Y = np.zeros((N,1))
	Y[:,0] = data[:, -1]
	return X, Y

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def GD(X, Y, W, N):
	G = np.zeros(dim)
	for n in range(0, N):
		s = sigmoid(-np.dot(W, X[n].transpose()) * Y[n][0])
		G += s * (-Y[n][0]) * X[n]
	return G/N

def SGD(X, Y, W, n):
	s = sigmoid(-np.dot(W, X[n].transpose()) * Y[n][0])
	G = s * (Y[n][0]) * X[n]
	return G

def error(X, W, Y, N):
	cnt = 0
	for i in range(0, N):
		H = np.dot(W, X[i].transpose())
		# print H
		if np.sign(H) != Y[i][0]:
			cnt += 1
	return float(cnt)/N

def Logistic(X_train, Y_train, N_train, X_test, Y_test, N_test):
	Ein = [None]*T
	Eout = [None]*T
	Ein_S = [None]*T
	Eout_S = [None]*T
	W = np.zeros(dim)
	# GD
	for i in range(0, T):
		G = GD(X_train, Y_train, W, N_train)
		W = W - lr * G
		Ein[i] = error(X_train, W, Y_train, N_train)
		Eout[i] = error(X_test, W, Y_test, N_test)
	# SGD
	W.fill(0)
	for i in range(0, T):
		G = SGD(X_train, Y_train, W, i%N_train)
		W = W + lr * G
		Ein_S[i] = error(X_train, W, Y_train, N_train)
		Eout_S[i] = error(X_test, W, Y_test, N_test)

	print "Ein =", Ein[0], "Eout =", Eout[0]
	print "Ein =", Ein[T-1], "Eout =", Eout[T-1]

	t = list(range(0, T))
	plt.xlabel('T')
	plt.ylabel('Ein')
	plt.plot(t, Ein, 'b', t, Ein_S, 'r')
	plt.legend(['GD', 'SGD'])
	plt.show()

	plt.xlabel('T')
	plt.ylabel('Eout')
	plt.plot(t, Eout, 'b', t, Eout_S, 'r')
	plt.legend(['GD', 'SGD'])
	plt.show()

X_train, Y_train = getdata("hw3_train.dat")
N_train , dim = X_train.shape
X_test, Y_test = getdata("hw3_test.dat")
N_test, dim = X_test.shape

Logistic(X_train, Y_train, N_train, X_test, Y_test, N_test)
