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
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.mllib.evaluation      import MulticlassMetrics

## Module Constants
APP_NAME = "MySparkApplication"

## Closure Functions
def tokenize(item):
    vector = Vectors.dense(float(item[0]),float(item[1]),float(item[2]),float(item[3]),float(item[4]),float(item[5]),float(item[8]),float(item[9]),float(item[10]),float(item[14]),float(item[15]),float(item[16]),float(item[17]),float(item[18]),float(item[19]))
    if item[20] == "20":
        label = 0.0
    elif item[20] == "21":
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
    print(iris_points.collect())
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
    splits = iris_points.randomSplit([0.6, 0.4], seed = 11)
    training = splits[0]
    testing = splits[1]

    """
    接下来，首先进行数据集的划分，这里划分60%的训练集和40%的测试集：
    然后，构建逻辑斯蒂模型，用set的方法设置参数，比如说分类的数目，这里可以实现多分类逻辑斯蒂模型：
    接下来，调用多分类逻辑斯蒂模型用的predict方法对测试数据进行预测，
    并把结果保存在MulticlassMetrics中。
    这里的模型全名为LogisticRegressionWithLBFGS，
    加上了LBFGS，表示Limited-memory BFGS。
    其中，BFGS是求解非线性优化问题（L(w)求极大值）的方法，
    是一种秩-2更新，以其发明者Broyden, Fletcher, Goldfarb和Shanno的姓氏首字母命名
    语法参考:https://spark.apache.org/docs/1.6.0/api/python/pyspark.mllib.html#pyspark.mllib.classification.LogisticRegressionWithLBFGS
    注意,在spark2.0.0里LogisticRegressionWithLBFGS会报如下错误
    TypeError: 'float' object cannot be interpreted as an integer
    这个错误正好在下一版解决了
    日志参考:https://issues.apache.org/jira/browse/SPARK-20862
        softmax回归模型，该模型是logistic回归模型在多分类问题上的推广，在多分类问题中，类标签 \textstyle y 可以取两>个以上的值。 Softmax回归模型对于诸如MNIST手写数字分类等问题是很有用的，该问题的目的是辨识10个不同的单个数字。Softmax回归是有监督的，不过后面也会介绍它与深度学习/无监督学习方法的结合。
        迭代次数是给梯度下降法用的.
    记住下面一点就可以了：sofmax使得logistic能多分类
        Sigmoid函数是MISO,Softmax就是MIMO的Sigmod函数
        就是把一堆实数的值映射到0-1区间，并且使他们的和为1。一般用来估计posterior probability，在多分类任务中有用到>。
        """
    model = LogisticRegressionWithLBFGS.train(training,iterations=120, numClasses=2)
    """
    这里，采用了test部分的数据每一行都分为标签label和特征features，
    然后利用map方法，对每一行的数据进行model.predict(features)操作，
    获得预测值。并把预测值和真正的标签放到predictionAndLabels中。
    我们可以打印出具体的结果数据来看一下：
    注意,预测出来的值必须转换成float,否则在MulticlassMetrics会出现int到double转换的错误
    TypeError: DoubleType can not accept object 1 in type <class 'int'>
    """
    predictionAndLabels = testing.map(lambda item: [float(model.predict(item.features)), item.label])
    print(predictionAndLabels.collect())
  metrics = MulticlassMetrics(predictionAndLabels)
    print("精确度:"+str(metrics.precision()))
    print("准确率:"+str(metrics.accuracy))
    print("召回率/:"+str(metrics.recall()))
    print("混淆矩阵")
    print(metrics.confusionMatrix())
    """
    precision(label=None)
    Returns precision or precision for a given label (category) if specified.
    召回率已经被取消了.
    https://spark.apache.org/docs/1.6.0/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.MulticlassMetrics
    在人工智能中，混淆矩阵（confusion matrix）是可视化工具，
    特别用于监督学习，在无监督学习一般叫做匹配矩阵。
    混淆矩阵的每一列代表了预测类别，
    每一列的总数表示预测为该类别的数据的数目，
    每一行代表了数据的真实归属类别，
    每一行的数据总数表示该类别的数据实例的数目
    因为是150的样本，使用了40%做测试集,本次随机出52个测试样本.
    下面的混淆矩阵,
    15:有15个第一类被预测为第一类
    第2行3列的1表示有1个第2类被预测为第三类类
    DenseMatrix([
    [ 15.,   0.,   0.],
    [  0.,  21.,   1.],
    [  0.,   1.,  14.]])
    为什么要提出混淆矩阵呢？是因为，如果预测精度很低，
    我想知道，到底是哪几个分类出问题了,从上面的矩阵可以看出，第2类和第3类互相混淆了，
    从这个矩阵，我可以知道下一步应该去优化哪几个分类的区分度，我可以改算法，改feature.
    """

    sc.stop()
if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    # Execute Main functionality
    main(sc)
