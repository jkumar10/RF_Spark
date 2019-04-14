from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

# filename = 'file:///N/u/jaikumar/data/'+str(sys.argv[1])
filename='file:///N/u/jaikumar/data/2006.csv'
spark=SparkSession.builder.appName('RF_air').getOrCreate()
data = spark.read.csv(filename,inferSchema=True,header=True)
df1=data.drop()
df2=df1.withColumn('isDelay',when(df1.DepDelay <= 0,'No').otherwise('Yes'))
assembler = VectorAssembler(
  inputCols=['Year',
             'Month',
             'DayofMonth',
             'DayOfWeek',
             'DepTime',
             'CRSDepTime',
             'ArrTime',
             'CRSArrTime',
             'FlightNum',
             'ActualElapsedTime',
             'CRSElapsedTime',
             'AirTime',
             'ArrDelay',
             'Distance',
             'TaxiIn',
             'TaxiOut',
             'CarrierDelay',
            'WeatherDelay',
            'SecurityDelay',
            '0SDelay',
            'LateAircraftDelay'],
              outputCol="features")

output = assembler.transform(df2)

indexer = StringIndexer(inputCol="isDelay", outputCol="isDelayIndex")

output_fixed = indexer.fit(output).transform(output)

final_data = output_fixed.select("features",'isDelayIndex')

train_data,test_data = final_data.randomSplit([0.3,0.7])

rfc = RandomForestClassifier(labelCol='isDelayIndex',featuresCol='features')

rfc_model = rfc.fit(train_data)

rfc_predictions = rfc_model.transform(test_data)

acc_evaluator = MulticlassClassificationEvaluator(labelCol="isDelayIndex", predictionCol="prediction", metricName="accuracy")

rfc_acc = acc_evaluator.evaluate(rfc_predictions)

print("Here are the results!")

print('A random forest ensemble had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))


