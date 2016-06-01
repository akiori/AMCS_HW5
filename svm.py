import numpy as np
import math
import pylab as pl

def IsZero(data):
	if data > 1e-5 or data < -1e-5:
		return False
	return True

def IsZeroVec(Vec):
	for i in range(len(Vec)):
		if not(IsZero(Vec[i])):
			return False
	return True

# solve Equality Constrained Quadratic Programming 
# f(x) = 1/2*d.T*G*d+r.T*d     s.t. A*d = b
def ECQP(G, r, A, b):
	rowA, colA = A.shape
	left = np.vstack((G, A))
	right = np.vstack((A.T, np.zeros((rowA, rowA))))

	x = np.linalg.solve(np.hstack((left, right)), np.vstack((-r, b)))

	solution = x[:colA]
	lamb = x[colA:]
	return solution, lamb
	
#active set method
#CONSTRAINT: Gx <= h Ax = b
def ASM(x0, hessian, r, G, h, A = [[0]], b = 0):
	#Step 1: find active set indices
	#Ax = b may be empty
	zeroIdx = []
	if A.shape[0] == len(x0):
		rowA, colA = A.shape
	else:
		rowA = 0
	rowG, colG = G.shape

	sum = np.dot(G, x0)
	tmp = sum - h
	for i in range(rowG):
		if IsZero(tmp[i]):
			zeroIdx.append(i)
	iter = 0

	x = x0
	while True and iter < 200:	
	#Step 2 solve equality constraint optimization
		rNew = hessian.dot(x)+r
		ANew = np.zeros((rowA+len(zeroIdx), colG))
		bNew = np.zeros((rowA+len(zeroIdx), 1))

		for i in range(rowA):
			for j in range(colG):
				ANew[i][j] = A[i][j]
		count = 0
		for i in zeroIdx:
			for j in range(colG):
				ANew[rowA+count][j] = G[i][j]
			count += 1
		d, lamb = ECQP(hessian, rNew, ANew, bNew)
		
		is_zero_vector = True
		if np.linalg.norm(d) > 1e-5:
			is_zero_vector = False
		if is_zero_vector:
		#if IsZeroVec(d):
		#for i in range(len(d)):
		#	if  np.linalg.norm(d) > 1e-5:
		#		is_zero_vector = False
		#		break
		#Step 3 if d is a zero vector
			isOpt = True
			idxMin = np.argmin(lamb)
			lambMin = lamb[idxMin]
			if lambMin < 0:
				isOpt = False

			if isOpt:
				output = np.zeros((rowG, 1))
				shift = 0
				for i in zeroIdx:
					output[i][0] = lamb[shift]
					shift += 1 
				return x, output
				break
			else:
				zeroIdx.remove(idxMin)
			iter += 1
		else:
		#Step 4 if not
			alpha = 1.
			idxMin = -1
			for i in range(rowG):
				if zeroIdx.count(i) == 0:
					k = (h[i][0]-G[i].T.dot(x))/(G[i].T.dot(d))
					if k <= alpha and G[i].dot(d) > 0:
						idxMin = i
						alpha  = k
			x = x+alpha*d
			if idxMin != -1:
				zeroIdx.append(idxMin)
			iter += 1
	return x, lamb


def KernelFunc(x,  y, sigma = 7.0):
	return np.exp(-np.linalg.norm(x-y)/ sigma)

	
def generator(numOfPoi = 70, mean1 = [2.5, 1.5],mean2 = [-1.0, -1.3],mean3 = [-6.0,-7.0],mean4 = [3.3, 9.5],cov = [[1.0,0.0],[0.0,1.0]]):
	x1 = np.vstack((np.random.multivariate_normal(mean1,cov,numOfPoi),np.random.multivariate_normal(mean2,cov,numOfPoi)))
	x2 = np.vstack((np.random.multivariate_normal(mean3,cov,numOfPoi),np.random.multivariate_normal(mean4,cov,numOfPoi)))
	y1 = np.ones(len(x1))
	y2 = np.ones(len(x2)) * -1
	return x1, y1, x2, y2

def partition(x1, y1, x2, y2):
	numSmp, samples_dim = x1.shape
	lenTrain = int(numSmp * 0.8)
	test_len = numSmp - lenTrain
	x_tra = np.vstack((x1[:lenTrain], x2[:lenTrain]))
	y_tra = np.hstack((y1[:lenTrain], y2[:lenTrain]))
	x_val = np.vstack((x1[lenTrain:], x2[lenTrain:]))
	y_val = np.hstack((y1[lenTrain:], y2[lenTrain:]))
	return x_tra, y_tra, x_val, y_val

def SVM(x, Y):
	numSmp, samples_dim = x.shape
	kerMat = np.zeros((numSmp, numSmp))
	hessian = np.zeros((numSmp, numSmp))
	A = np.zeros((1, numSmp))
	for i in range(numSmp):
		A[0][i] = Y[i]
		for j in range(numSmp):
			kerMat[i][j] = KernelFunc(x[i], x[j])
			#kerMat[i][j] = np.dot(x[i], x[j])
			hessian[i][j] = kerMat[i][j] * Y[i] * Y[j]
	#print(kerMat)
	#print(hessian)
	r = np.ones((numSmp, 1)) * -1
	b = np.zeros((1, 1))
	G = np.diag(np.ones(numSmp) * -1)
	h = np.zeros((numSmp, 1))
	count = 0
	for i in range(numSmp):
		if (Y[i] == 1):
			count += 1
	x0 = np.ones((numSmp, 1))
	for i in range(numSmp):
		if Y[i] == 1:
			#x0[i][0] = 1./count
			x0[i][0] = numSmp - count
		else:
			#x0[i][0] = 1./(numSmp-count)
			x0[i][0] = count
	solution,  lamb = ASM(x0, hessian, r, G, h, A, b)
	a = np.ravel(solution)
	#print(a)
	# find  ECQP multipliers larger than zero,  Support vectors have non zero lagrange multipliers
	sv = a > 1e-5
	ind = np.arange(len(a))[sv]
	print(ind)
	alpha = a[sv]
	suppData = x[sv]
	suppLab = Y[sv]
	print("Altogether "+ str(numSmp) +" points with "+ (str(len(alpha))) +" support vectors")
	# Intercept
	intercept = 0
	for i in range(len(alpha)):
		intercept += suppLab[i]
		intercept -= np.sum(alpha * suppLab * kerMat[ind[i], sv])
	intercept /= len(alpha)
	return alpha, suppData, suppLab, intercept

def project(x, alpha, suppData, suppLab, intercept):
	y_predict = np.zeros(len(x))
	for i in range(len(x)):
		s = 0
		for a,  data,  y in zip(alpha,  suppData,  suppLab):
			s += a * y * KernelFunc(x[i],  data)
		y_predict[i] = s
	return (y_predict + intercept)

def plot(x1_train, x2_train, alpha, suppLab, suppData, grid_half_len = 8):
	pl.plot(x1_train[:, 0], x1_train[:, 1], "r^")
	pl.plot(x2_train[:, 0], x2_train[:, 1], "g+")
	pl.scatter(suppData[:, 0], suppData[:, 1],  s=10,  c="k")
	x1, x2 = np.meshgrid(np.linspace(-grid_half_len, grid_half_len, 100), np.linspace(-grid_half_len, grid_half_len, 100))
	x = np.array([[x1,  x2] for x1, x2 in zip(np.ravel(x1), np.ravel(x2))])
	Z = project(x , alpha, suppData, suppLab, intercept).reshape(x1.shape)
	pl.contour(x1, x2, Z, [0.0], colors='k', linewidths=1, origin='lower')
	pl.contour(x1, x2, Z + 1, [0.0], colors='y', linewidths=1, origin='lower')
	pl.contour(x1, x2, Z - 1, [0.0], colors='y', linewidths=1, origin='lower')
	#pl.axis("tight")
	pl.show()

def predict(x, alpha, suppData, suppLab, intercept):
	return np.sign(project(x, alpha, suppData, suppLab, intercept))


if __name__ == "__main__":
	x1,  y1,  x2,  y2 = generator()
	#x_tra,  y_tra = split_train(x1,  y1,  x2,  y2)
	#x_val,  y_val = split_test(x1,  y1,  x2,  y2)
	x_tra, y_tra, x_val, y_val = partition(x1, y1, x2, y2)
	alpha, suppData, suppLab, intercept = SVM(x_tra,  y_tra)
	Y_predict = predict(x_val, alpha, suppData, suppLab, intercept)
	precision = np.sum(Y_predict == y_val) / len(y_val)
	print("The precision is " + str(precision))
	plot(x_tra[y_tra==1],  x_tra[y_tra==-1],  alpha,  suppLab,  suppData, 15)
