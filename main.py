import csv
import random
import math

def loadData(filename):
    lines = csv.reader(open(filename,"r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = ([float(a) for a in dataset[i]])
    return dataset

def splitDataset(dataset,trainRadio):
    testSet = []
    trainSet = list(dataset)
    testNum = int((1-trainRadio)*len(dataset))
    for i in range(testNum):
        index = random.randrange(len(trainSet))
        testSet.append(trainSet.pop(index))
    return [trainSet,testSet]

def separateByClass(dataSet):
    separated = {}
    for i in range(len(dataSet)):
        if dataSet[i][-1] not in separated:
            separated[dataSet[i][-1]] = []
        separated[dataSet[i][-1]].append(dataSet[i])
    return separated

def mean(numbers):  #list or dict
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    numerator = sum([pow(avg-x,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(numerator)

def summarize(dataSet):
    eachAttribute = list(zip(*dataSet))
    del eachAttribute[-1]
    return [(mean(attribute),stdev(attribute)) for attribute in eachAttribute]

def summarizeByClass(dataSet):
    separated = separateByClass(dataSet)
    summaries = {}
    for classValue,instance in separated.items():
        summaries[classValue] = summarize(instance)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent



def calculateClassProbability(summaries,inputVector):
    probabilities = {}
    for classValue,classSummaries in summaries.items():
        probabilities[classValue] = 1.0
        for i in range(len(classSummaries)):
            mean,stdev = classSummaries[i]
            probabilities[classValue] *= calculateProbability(inputVector[i],mean,stdev)
    return probabilities



def predict(summaries,inputVector):
    probabilities = calculateClassProbability(summaries,inputVector)
    bestClass,bestProbability = None,-1
    for classValue,probabilityValue in probabilities.items():
        if classValue is None or bestProbability < probabilityValue:
            bestClass = classValue
            bestProbability = probabilityValue
    return bestClass

def getPredictions(summaries,testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries,testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet,prediction):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == prediction[i]:
            correct += 1
    return correct/float(len(testSet)) * 100.0


def main():
    filename = 'pima-indians-diabetes.data.csv'
    splitRatio = 0.67
    dataset = loadData(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(dataset)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))


main()