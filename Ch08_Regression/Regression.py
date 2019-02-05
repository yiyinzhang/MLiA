import numpy as np

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		lineArr = []
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat
	
def standRegres(xArr, yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xTx = xMat.T * xMat
	if np.linalg.det(xTx) == 0.0:
		print("This matrix is singular, cannot do inverse")
		return
	ws = xTx * (xMat.T * yMat)
	return ws
	
def lwlr(testPoint, xArr, yArr, k = 2.0):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	m = np.shape(xMaT)[0]
	weights = np.eye((m))
	for j in range(m):
		diffMat = testPoint - xMat[j, :]
		weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k ** 2))
	xTx = xMat.T * (weights * xMat)
	if np.linalg.det(xTx) == 0.0:
		print("This matrix is singular, cannot do inverse")
		return
	ws = xTx * (xMat * (weights * yMat))
	return testPoint * ws
	
def lwlrTest(testArr, xArr, yArr, k = 0.2):
	m = np.shape(testArr)[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i], xArr, yArr, k)
	return yHat
	
def ridgeRegres(xMat, yMat, lam = 0.2):
	xTx = xMat.T * xMat
	denom = xTx + lam * np.eye(np.shape(xMat)[1])
	if np.linalg.det(denom) == 0.0:
		print("This matrix is singular, cannot do inverse")
		return
	ws = denom.I * (xMat.T * yMat)
	return ws
	
def ridgeTest(xArr, yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xMeans = np.mean(xArr, 0)
	xVar = np.var(xArr, 0)
	yMean = np.mean(yArr, 0)
	xMat = (xMat - xMeans)/xVar
	yMat = yMat - yMean
	m, n = np.shape(xMat)
	numTestPts = 30
	wMat = np.zeros((numTestPts, n))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat, yArr, np.exp(i - 10))
		wMat[i, :] = ws.T
	return wMat
	
def rssError(yArr, yHatArr):
	return ((yArr - yHatArr) ** 2).sum()
	
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xMat = np.regularize(xMat)
	yMean = np.mean(yMat, 0)
	yMat = yMat - yMean
	m, n = np.shape(xMat)
	returnMat = np.zeros((numIt, n))
	ws = np.zeros((n, 1))
	wsTest = ws.copy()
	wsMax = ws.copy()
	for i in range(numIt):
		print(ws.T)
		lowestError = inf
		for j in range(n):
			for sign in [-1, 1]:
				wsTest = ws.copy()
				wsTest[j] += eps * sign
				yTest = xMat * wsTest
				rssE = rssError(yMat.A, yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i, :] = ws.T
	return returnMat
	
def crossValidation(xArr, yArr, numVal = 10):
	m = len(xArr)
	indexList = range(m)
	errorMat = np.zeros((numVal, 30))
	for i in range(numVal):
		trainX = []
		trainY = []
		testX = []
		testY = []
		np.random.shuffle(indexList)
		for j in range(m):
			if j < 0.9 * m:
				trainX.append(xArr[indexList[j]])
				trainY.append(yArr[indexList[j]])
			else:
				testX.append(xArr[indexList[j]])
				testY.append(xArr[indexList[j]])
		wMat = ridgeTest(xArr, yArr)
		for k in range(30):
			matTrainX = np.mat(trainX)
			matTestX = np.mat(testX)
			meanTrain = np.mean(matTrainX, 0)
			varTrain = np.var(matTrainX, 0)
			matTestX = (matTestX - meanTrain)/varTrain
			yEst = matTestX * np.mat(wMat[i, :]).T + np.mean(trainY, 0)
			errorMat[i, k] = ((yEst.A - np.array(testY)) ** 2).sum()
	meanErrors = np.mean(errorMat, 0)
	minMean = float(np.min(meanErrors))
	bestWeights = wMat[np.nonzero(meanErrors == minMean)]
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	meanX = np.mean(xMat, 0)
	varX = np.var(xMat, 0)
	unReg = bestWeights/varX
	print("the best model from ridge regression is:\n", unReg)
	print("the constant term is:" -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat, 0))
	