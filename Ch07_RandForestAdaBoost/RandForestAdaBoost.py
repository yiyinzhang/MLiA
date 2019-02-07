import numpy as np
import random

def loadSimpData():
	dataArr = np.array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
	labelArr = [1.0, 1.0, -1.0, -1.0, 1.0]
	return dataArr, labelArr
	
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline.split('\t')) - 1
	dataArr = []
	labelArr = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataArr.append(lineArr)
		labelArr.append(float(curLine[-1]))
	return dataArr, labelArr
	
def stumpClassify(dataMat, dim, threshVal, threshIneq):
	retArray = []
	if threshIneq == 'lt':
		retArray[dataMat[:, dim] <= threshVal] = -1.0
	else:
		retArray[dataMat[:, dim] > threshVal] = -1.0
	return retArray
	
def buildStump(dataArr, labelArr, D):
	dataMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).T
	m, n = np.shape(dataMat)
	numSteps = 10.0
	bestStump = {}
	bestClasEst = np.zeros((m, 1))
	minError = inf
	for i in range(n):
		rangeMin = dataMat[:, i].min()
		rangeMax = dataMat[:, i].max()
		stepSize = (rangeMax - rangeMin)/numSteps
		for j in range(-1, int(numSteps) + 1):
			for inequal in ['lt', 'gt']:
				threshVal = rangeMin + float(j) * stepSize
				predictedVals =stumpClassify(dataMat, i, threshVal, inequal)
				errArr = np.ones((m, 1))
				errArr[predictedVals ==labelMat] = 0
				weightedError = D.T * errArr
				if weightedError < minError:
					bestClasEst = predictedVals.copy()
					minError = weightedError
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump, minError, bestClasEst
	
def adaBoostTrainDS(dataArr, labelArr, numIt = 40):
	weakClassArr = []
	m = np.shape(dataArr)
	