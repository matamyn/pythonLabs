import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

majorLocator = MultipleLocator(1)

mpl.rcParams['figure.figsize']=(20,6)
np.set_printoptions(threshold=np.nan)
dataset = []
def readDataset(filename):
    infile = open(filename, 'r')
    import csv
    for row in csv.reader(infile):
        dataset.append(row)
    infile.close()
    for j in range (len(dataset[0])):
        try:
            float(dataset[0][j])
        except ValueError:
            continue
        else:
            for i in range(len(dataset)):
                dataset[i][j] =float(dataset[i][j]) 
    

def featureSelect(begin,number,*feature):
    k=0
    if (number=="all"): number = len(dataset) - begin
    featureMatr = np.empty((number, len(feature)))
    for j in feature:
         for i in range(begin,begin+number):
            featureMatr[i][k] = dataset[i][j]
    k+=1     
    return featureMatr

def getVanderMatr(featureMatr,degree):
    vanderMatr = []
    for j in range (len(featureMatr)):
        arr=[]
        for i in range(degree+1):
            arr.append(pow(featureMatr[j][0],i))
        vanderMatr.append(arr)    
    vanderMatr = np.asarray(vanderMatr)
    return vanderMatr

def oneDimensionRegress(vanderMatr, responseMatr, lambdaR):
    matrAA = np.dot(vanderMatr.transpose(),vanderMatr) + lambdaR*np.eye(len(vanderMatr[0]))
    theta1 = np.linalg.inv(matrAA)
    theta2 = np.dot(vanderMatr.transpose(), responseMatr)
    theta = np.dot(theta1,theta2)
    return theta, np.linalg.det(matrAA), np.linalg.cond(matrAA)

def getResponseMatr(begin,number, column):
    if (number=="all"): number = len(dataset) - begin
    responseMatr = np.array([t_arr[column] for t_arr in dataset])
    responseMatr = responseMatr.reshape((len(responseMatr),1))
    result = responseMatr[begin:(begin+number)]
    return result

def slidingwindow(width, degree, lambdaR):
    predictArray = []
    errorsArray = []
    featureMatr = featureSelect(0,"all",0)
    responseMatr = getResponseMatr(0,"all",1)
    for begin in range(len(featureMatr) - width - 1):
        end = begin + width
        next = end + 1
        shortFeatureMatr = featureMatr[begin:end,:]
        vanderMatr = getVanderMatr(shortFeatureMatr,degree)
        shortResponseMatr = responseMatr[begin:end]
        [theta,det,cond] = oneDimensionRegress(vanderMatr, shortResponseMatr, lambdaR)
        
        predictFeatureMatr = featureMatr[next,:].reshape(1,1)
        predictVanderMatr = getVanderMatr(predictFeatureMatr,degree)
        predict = np.dot(predictVanderMatr,theta)
        predictArray.append(predict)
   
        errorsArray.append(responseMatr[next] - predict)
    errorsArray = (np.asarray(errorsArray)).reshape(len(errorsArray),1)
    predictArray = (np.asarray(predictArray)).reshape(len(predictArray),1)
    sse = np.dot(errorsArray.transpose(), errorsArray)
    mse = sse/(width - 2)
    #plotDots(featureMatr,responseMatr, predictArray)
    return mse,math.fabs(det),cond

def plotDots(featureMatr, responseMatr, predicted):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(featureMatr[101:],responseMatr[101:],s=2)
    ax.scatter(featureMatr[101:], predicted, s=2) 
    print(featureMatr[101:].shape,(predicted).shape)

    plt.xlabel(r'$x$') 
    plt.grid(True) 
    plt.show()
    
def plotData(arr, name):
    ax = plt.subplot()
    # plt.semilogy([x for x in range (1, len(arr)+1)], arr, marker = ".", markersize = 8, markeredgewidth = 5)
    plt.plot([x for x in range (1, len(arr)+1)], arr, marker = ".", markersize = 8, markeredgewidth = 5)
    ax.xaxis.set_major_locator(majorLocator)
    plt.xlabel(r'$degree$')
    plt.ylabel(name)
    plt.grid(True) 
    plt.show()
    
def mseOnDegree(min,max,width, lambdaR):
    mseArray = []
    detArray = []
    condArray = []
    for degree in range(min,max + 1):
        [mse,det,cond] = slidingwindow(width,degree,lambdaR)
        mseArray.append(mse)
        detArray.append(det)
        condArray.append(cond)
    mseArray = np.asarray(mseArray).reshape(len(mseArray),)
    detArray = np.asarray(detArray).reshape(len(detArray),)
    condArray = np.asarray(condArray).reshape(len(condArray),)
    print("MSE",mseArray)
    print("determ",detArray)
    print("cond",condArray)
    plotData(mseArray, "MSE")
    plotData(detArray, "Determinant")
    plotData(condArray, "Condition Number")

def main():
   
    readDataset('temp_norm.csv')
    mseOnDegree(1,15,100,1)
    #slidingwindow(100,3,0)
if __name__ == "__main__":
    main()
	