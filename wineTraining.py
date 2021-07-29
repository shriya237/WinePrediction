import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql.functions import col, expr, when
import numpy as np
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel


conf = pyspark.SparkConf().setAppName('WineQuality')
sc = pyspark.SparkContext(conf=conf).getOrCreate()
spark = SparkSession(sc)

schema = StructType([
    StructField("fixed acidity", DoubleType(), True),
    StructField("volatile acidity", DoubleType(), True),
    StructField("citric acid", DoubleType(), True),
    StructField("residual sugar", DoubleType(), True),
    StructField("chlorides", DoubleType(), True),
    StructField("free sulfur dioxide", DoubleType(), True),
    StructField("total sulfur dioxide", DoubleType(), True),
    StructField("density", DoubleType(), True),
    StructField("pH", DoubleType(), True),
    StructField("sulphates", DoubleType(), True),
    StructField("alcohol", DoubleType(), True),
    StructField("quality", DoubleType(), True)
])


train_df = spark.read.format("csv").load("TrainingDataset.csv" , header = True ,sep =";")

train_df.printSchema()

for col_name in train_df.columns[1:-1]+['""""quality"""""']:
    train_df = train_df.withColumn(col_name, col(col_name).cast('float'))
train_df = train_df.withColumnRenamed('""""quality"""""', "label")

#getting the features and label seperately and converting it to numpy array
features =np.array(train_df.select(train_df.columns[1:-1]).collect())
label = np.array(train_df.select('label').collect())

#creating the feature vector
VectorAssembler = VectorAssembler(inputCols = train_df.columns[1:-1] , outputCol = 'features')
df_tr = VectorAssembler.transform(train_df)
df_tr = df_tr.select(['features','label'])

def to_labeled_point(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)


#rdd converted dataset
dataset = to_labeled_point(sc, features, label)


#Splitting the dataset into train and test
training, test = dataset.randomSplit([0.7, 0.3],seed =11)


#Creating a random forest training classifier
RFmodel = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=30, maxBins=32)


#predictions
predictions = RFmodel.predict(test.map(lambda x: x.features))


#getting a RDD of label and predictions
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)


testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())    
print('Test Error = ' + str(testErr))


#save training model
RFmodel.save(sc, 's3://trainingmodel.model')
