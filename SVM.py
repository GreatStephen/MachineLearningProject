
from pyspark import SparkConf, SparkContext
#from pyspark import mllib.linalg.Vector
#from pyspark import mllib.stat.{MultivariateStatisticalSummary, Statistics}
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Vector
from pyspark.mllib.stat import MultivariateStatisticalSummary
from pyspark.mllib.stat import Statistics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree                import DecisionTree
from pyspark.mllib.evaluation      import MulticlassMetrics
from pyspark.mllib.classification import SVMModel
from pyspark.mllib.classification import SVMWithSGD

## Module Constants
APP_NAME = "MySparkApplication"

## Closure Functions
def tokenize(item):
    vector = Vectors.dense(float(item[0]),float(item[1]),float(item[2]),float(item[3]),float(item[4]),float(item[5]),float(item[6]),float(item[7]),float(item[8]),float(item[9]),float(item[10]),float(item[11]),float(item[12]),float(item[13]),float(item[14]),float(item[15]),float(item[16]),float(item[17]),float(item[18]),float(item[19]))
    if item[20] == "20": #label=no
        label = 0.0
    elif item[20] == "21": #label=yes
        label = 1.0
    else:
        label = 2.0

    item = LabeledPoint(label,vector)
    return item

## Main functionality

def main(sc):
    bank_lines = sc.textFile("bank5.csv")
    bank_lines = bank_lines.map(lambda item: item.split(","))
    #print(bank_lines.collect())
    """
    我们对把每一行都通过split函数形成数组.整个iris_lines看上去是一个二维数组.
    [
   i     ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa'],
        ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa'],
        ['4.7', '3.2', '1.3', '0.2', 'Iris-setosa'],
    ]
    """
    bank_points = bank_lines.map(lambda item:tokenize(item))
    print(bank_points)
 #print(iris_points.collect())
   # print(iris_points)
    """
    通过map把rdd中每一项都转换成一个labeledpoint,
    [
        LabeledPoint(0.0, [5.1,3.5,1.4,0.2]),
        LabeledPoint(0.0, [4.9,3.0,1.4,0.2]),
        LabeledPoint(0.0, [4.7,3.2,1.3,0.2]),
        LabeledPoint(0.0, [4.6,3.1,1.5,0.2]),
        LabeledPoint(0.0, [5.0,3.6,1.4,0.2]),
        LabeledPoint(0.0, [5.4,3.9,1.7,0.4]),
    ]
    """
  splits = bank_points.randomSplit([0.5, 0.5],11)
    training = splits[0]
    testing = splits[1]
   # print(training.count())
   # print(testing.count())

   #model = DecisionTree.trainClassifier(training, 3,{},’gini’,maxDepth=20)
    numIteration3 =100
    stepsize =1
    miniBatchFraction=1
    model = SVMWithSGD.train(training,numIteration3,stepsize,miniBatchFraction)
    print(model)
    predictionAndLabel = training.map(lambda bank_points:(bank_points.label,model.predict(bank_points.features)))
    trainErr = predictionAndLabel.filter(lambda predictionAndLabel:predictionAndLabel[0]!=predictionAndLabel[1]).count()/float(bank_points.count())
    print ("Accuracy = "+str(1-float(trainErr)))
  #  prediction = model.predict(testing.map(bank_points.features))
  #  predictionAndLabel = prediction.zip(test.map(bank_points.label))
    model.save(sc,"model")
    sameModel =SVMModel.load(sc,"model")
    err= sameModel.predict(bank_points.collect()[0].features)
    sc.stop()

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    # Execute Main functionality
    main(sc)
