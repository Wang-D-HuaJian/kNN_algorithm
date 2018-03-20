import sys
sys.path.append("F:/Machine_Learning_Algorithm")
from kNN_algorithm.kNN import *
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
ax.axis([-2,25,-0.2,2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()

#normMat, ranges, minValues = autoNorm(datingDataMat)
#print("normMat:\n" + str(normMat) + " \nranges:\n" + str(ranges) + "\nminValues\n" + str(minValues))

#datingClassTest()
#classifyPerson()

#############测试数据
testVector = img2vector('testDigits/0_13.txt')
print(testVector[0,0:31])
print(testVector[0,32:63])

handwritingClassTest()



