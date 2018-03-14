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
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
## Module Constants
APP_NAME = "MySparkApplication"

## Closure Functions
def tokenize(item):
    vector = Vectors.dense(float(item[0]),float(item[1]),float(item[2]),float(item[3]),float(item[4]),float(item[5]),float(item[7]),float(item[8]),float(item[9]),float(item[10]),float(item[14]),float(item[15]),float(item[16]),float(item[17]),float(item[18]),float(item[19]))
    if item[20] == "20":
        label = 0.0
    else:
        label = 1.0

    item = LabeledPoint(label,vector)
    return item

def vectored(item):
    vector = Vectors.dense(float(item[0]),float(item[1]),float(item[2]),float(item[3]),float(item[4]),float(item[5]),float(item[7]),float(item[8]),float(item[9]),float(item[10]),float(item[14]),float(item[15]),float(item[16]),float(item[17]),float(item[18]),float(item[19]))
    return vector


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
 iris_vector = iris_lines.map(lambda item:vectored(item))
    print(iris_vector)
    summary = Statistics.colStats(iris_vector)
    max_vector = summary.max()
    min_vector = summary.min()
    print("max " + str(summary.max()))
    print("min " + str(summary.min()))
    def scale(item):
        item0 = float(item[0])
        item1 = float(item[1])
        item2 = float(item[2])
        item3 = float(item[3])
        item4 = float(item[4])
        item5 = float(item[5])
        item6 = float(item[6])
        ##item7 = float(item[7])
        item8 = float(item[8])
        item9 = float(item[9])
        item10 = float(item[10])
        item11 = float(item[11])
        ##item12 = float(item[12])
        ##item13 = float(item[13])
        ##item14 = float(item[14])
        item15 = float(item[15])
        item16 = float(item[16])
        item17 = float(item[17])
        item18 = float(item[18])
        item19 = float(item[19])
        fitem0 = (item0 - min_vector[0])/(max_vector[0] - min_vector[0])
        fitem1 = (item1 - min_vector[1])/(max_vector[1] - min_vector[1])
        fitem2 = (item2 - min_vector[2])/(max_vector[2] - min_vector[2])
        fitem3 = (item3 - min_vector[3])/(max_vector[3] - min_vector[3])
        fitem4 = (item4 - min_vector[4])/(max_vector[4] - min_vector[4])
        fitem5 = (item5 - min_vector[5])/(max_vector[5] - min_vector[5])
        fitem6 = (item6 - min_vector[6])/(max_vector[6] - min_vector[6])
        ##fitem7 = (item7 - min_vector[7])/(max_vector[7] - min_vector[7])
        fitem8 = (item8 - min_vector[8])/(max_vector[8] - min_vector[8])
        fitem9 = (item9 - min_vector[9])/(max_vector[9] - min_vector[9])
        fitem10 = (item10 - min_vector[10])/(max_vector[10] - min_vector[10])
        fitem11 = (item11 - min_vector[11])/(max_vector[11] - min_vector[11])
        ##fitem12 = (item12 - min_vector[12])/(max_vector[12] - min_vector[12])
        ##fitem13 = (item13 - min_vector[13])/(max_vector[13] - min_vector[13])
        ##fitem14 = (item14 - min_vector[14])/(max_vector[14] - min_vector[14])
        fitem15 = (item15 - min_vector[15])/(max_vector[15] - min_vector[15])
        fitem16 = (item16 - min_vector[16])/(max_vector[16] - min_vector[16])
        fitem17 = (item17 - min_vector[17])/(max_vector[17] - min_vector[17])
        fitem18 = (item18 - min_vector[18])/(max_vector[18] - min_vector[18])
        fitem19 = (item19 - min_vector[19])/(max_vector[19] - min_vector[19])
        strval = '%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2,%s' % (fitem0,fitem1,fitem2,fitem3,fitem4,fitem5,fitem6,fitem8,fitem9,fitem10,fitem11,fitem15,fitem16,fitem17,fitem18,fitem19, item[20])
        return strval
    iris_lines = iris_lines.map(lambda item:scale(item))

    iris_lines = iris_lines.map(lambda item: item.split(","))
    for iris_line_item in iris_lines.collect():
        print(iris_line_item)
    iris_vector = iris_lines.map(lambda item:vectored(item))
    
  """
    examples/src/main/python/mllib/k_means_example.py"
    K-means也是聚类算法中最简单的一种。
    聚类的目的是找到每个样本x潜在的类别y，并将同类别y的样本x放在一起。
    比如上面的星星，聚类后结果是一个个星团，星团里面的点相互距离比较近，星团间的星星距离就比较远了。
    拿星团模型来解释就是要将所有的星星聚成k个星团，
    首先随机选取k个宇宙中的点（或者k个星星）作为k个星团的质心C，
    然后第一步对于每一个星星计算其到k个质心中每一个的距离，
    然后选取距离最近的那个星团作为C，
    这样经过第一步每一个星星都有了所属的星团；
    第二步对于每一个星团，重新计算它的质心u（对里面所有的星星坐标求平均）。
    重复迭代第一步和第二步直到质心不变或者变化很小。

    K-means面对的第一个问题是如何保证收敛，
    前面的算法中强调结束条件就是收敛，
    可以证明的是K-means完全可以保证收敛性。
    J函数= 每个点到质心距离平方和。
    J函数表示每个样本点到其质心的距离平方和。
    K-means是要将J调整到最小。
    假设当前J没有达到最小值，那么首先可以固定每个类的质心u，调整每个样例的所属的类别C来让J函数减少，
    同样，固定C，调整每个类的质心u也可以使J减小。
    这两个过程就是内循环中使J单调递减的过程。
    当J递减到最小时，C和u也同时收敛。
    （在理论上，可以有多组不同的C和u值能够使得J取得最小值，但这种现象实际上很少见）。

    由于畸变函数J是非凸函数，意味着我们不能保证取得的最小值是全局最小值，
    也就是说k-means对质心初始位置的选取比较感冒，
    但一般情况下k-means达到的局部最优已经满足需求。
    但如果你怕陷入局部最优，那么可以选取不同的初始值跑多遍k-means，
    然后取其中最小的J对应的C和u输出。
    """
    #Build the model (cluster the data)
    """
    如果iteration设为100，10，效果都不好
    但设为200,效果就好了
    """
    model = KMeans.train(iris_vector, 2, maxIterations=1000, initializationMode="random")
    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = model.centers[model.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    #Evaluate clustering by computing Within Set Sum of Squared Errors
    WSSSE = iris_vector.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))

    # Shows the result.Get the cluster centers, represented as a list of NumPy arrays.
    # Return the K-means cost (sum of squared distances of points to their nearest center) for this model on the given data.
    # Return the K-means cost (sum of squared distances of points to their nearest center) for this model on the given data.
    # predict(x)
    # Parameters:    x – A data point (or RDD of points) to determine cluster index.
    # Returns:    Predicted cluster index or an RDD of predicted cluster indices if the input is an RDD.
    # Find the cluster that each of the points belongs to in this model.
    print("model.k:"+str(model.k))
    centers = model.clusterCenters
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    iris_points = iris_lines.map(lambda item:tokenize(item))
    iris_compare= iris_points.map(lambda item:(item.label,model.predict(item.features)))
    for iris_comp_item in iris_compare.collect():
        print(iris_comp_item)
    # Save and load model
    #model.save(sc, "KMeansModel")
    #sameModel = KMeansModel.load(sc, "KMeansModel")
    
    
    sc.stop()


if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    # Execute Main functionality
    main(sc)
