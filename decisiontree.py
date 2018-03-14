## Spark Application - execute with spark-submit：
#spark-submit app.py
## Imports
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

## Module Constants
APP_NAME = "MySparkApplication"

## Closure Functions
def tokenize(item):
    vector = Vectors.dense(float(item[0]),float(item[1]),float(item[2]),float(item[3]),float(item[4]),float(item[5]),float(item[6]),float(item[10]),float(item[14]),float(item[15]),float(item[15]),float(item[16]),float(item[17]),float(item[18]))
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
    iris_lines = sc.textFile("bank5.csv")
    #print(iris_lines.collect())
    """
    Iris数据集也称鸢尾花卉数据集，是一类多重变量分析的数据集，是常用的分类实验数据集，
    由Fisher于1936收集整理。数据集包含150个数据集，分为3类，每类50个数据，
    每个数据包含4个属性。
    可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性
    预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类
    整个文件读取成一个一维数组，每行都是一个RDD
    [
        '5.1,3.5,1.4,0.2,Iris-setosa',
        '4.9,3.0,1.4,0.2,Iris-setosa',
        '4.7,3.2,1.3,0.2,Iris-setosa',
        '4.6,3.1,1.5,0.2,Iris-setosa',
    ]
    """
    iris_lines = iris_lines.map(lambda item: item.split(","))
    #print(iris_lines.collect())
    """
    我们对把每一行都通过split函数形成数组.整个iris_lines看上去是一个二维数组.
    [
        ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa'],
        ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa'],
        ['4.7', '3.2', '1.3', '0.2', 'Iris-setosa'],
    ]
    """
    iris_points = iris_lines.map(lambda item:tokenize(item))
    #print(iris_points.collect())
    print(iris_points)
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
    splits = iris_points.randomSplit([0.7, 0.3],11)
    training = splits[0]
    testing = splits[1]
    print(training.count())
    print(testing.count())
    """
    首先进行数据集的划分，这里划分70%的训练集和30%的测试集：
    然后，调用决策树的trainClassifier方法构建决策树模型，设置参数，比如分类数、信息增益的选择、树的最大深度等
    """
    #model = DecisionTree.trainClassifier(iris_points, numClasses = 3,maxDepth = 5,maxBins = 32,{})
    model = DecisionTree.trainClassifier(training, 3,{},maxDepth=13)
    print(model)
    print(model.toDebugString())
    """
    从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，
    由该特征的不同取值建立子结点，再对子结点递归地调用以上方法，构建决策树；
    直到所有特征的信息增均很小或没有特征可以选择为止，最后得到一个决策树。
    决策树需要有停止条件来终止其生长的过程。
    一般来说最低的条件是：当该节点下面的所有记录都属于同一类，或者当所有的记录属性都具有相同的值时。
    这两种条件是停止决策树的必要条件，也是最低的条件。
    在实际运用中一般希望决策树提前停止生长，限定叶节点包含的最低数据量，
    以防止由于过度生长造成的过拟合问题。

    这里，采用了test部分的数据每一行都分为标签label和特征features，
    然后利用map方法，对每一行的数据进行model.predict(features)操作，
    获得预测值。并把预测值和真正的标签放到predictionAndLabels中。
    我们可以打印出具体的结果数据来看一下：
    注意,预测出来的值必须转换成float,否则在MulticlassMetrics会出现int到double转换的错误
    TypeError: DoubleType can not accept object 1 in type <class 'int'>
    """
    #predict_result = model.predict([5.4,3.9,1.7,0.4])
    #print(predict_result)
    #下面是方法1,会出例外.
    #predictionAndLabels = testing.map(lambda item: [model.predict(item.features), item.label])
    #下面是方法2
    predictionList = []
    for item in testing.collect():
        predictionList.append([model.predict(item.features),item.label])
    predictionAndLabels = sc.parallelize(predictionList)
    print(predictionAndLabels.collect())
 metrics = MulticlassMetrics(predictionAndLabels)
    print("精确度:"+str(metrics.precision()))
    print("准确率:"+str(metrics.accuracy))
    print("召回率:"+str(metrics.recall()))
    print("混淆矩阵")
    print(metrics.confusionMatrix())
    """
    最后，我们把模型预测的准确性打印出来：
    """
    model.save(sc,"model")
    sc.stop()

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    # Execute Main functionality
    main(sc)
