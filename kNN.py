from numpy import *
import operator
from os import listdir

#####简单的带有标签的数据集
# def creatDataSet():
#    """创建数据训练集和对应标签"""
#    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
#    labels = ['A','A','B','B']
#    return group, labels

#####原始的k-近邻分类算法
def classify0(inX, dataSet, labels, k):
    """k-近邻算法"""
    dataSetSize = dataSet.shape[0]#返回训练集中的样本数目   ##shape函数返回矩阵或者数组的维数
    #############距离计算（欧式距离公式）
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    ####按距离从小到大排序（numpy的argsort排序算法）
    sortedDistIndicies = distances.argsort()

    #######选择距离最小的k个点#
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    #############排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

#####准备数据，将文本改成分类器可以接受的格式（矩阵形式）
def file2matrix(filename):
    """将给定文本改为为分类器可以接受的格式"""
    ####打开文件
    fr = open(filename)
    ###得到文件行数
    arrayOLines = fr.readlines()######将文件按行存储
    numberOfLines = len(arrayOLines)
    ####创建以零填充的矩阵NumPy
    returnMat = zeros((numberOfLines,3))

    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()######截断所有的回车字符
        listFromLine = line.split('\t')#####将上面得到的数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3]###取前三个元素，存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))######存储最后一列到classLabelVector（必须声明为整型）
        index += 1#######循环
    return returnMat, classLabelVector

######准备数据，对数据进行归一化处理，将所有特征值都转化到[0-1]区间内
def autoNorm(dataSet):
    """归一化特征值:newValue=(oldValue-min)/(max-min)"""
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

#####测试算法：作为完整程序验证分类器
def datingClassTest():
    """分类器针对约会网站的测试代码"""
    hoRatio = 0.01
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" %
            (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

#########使用算法：构建完整可用系统
def classifyPerson():
    """约会网站预测函数"""
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per years?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

def img2vector(filename):
    """将图像转换为测试向量"""
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))


######调用函数，检查结果
#classifyPerson()
handwritingClassTest()





