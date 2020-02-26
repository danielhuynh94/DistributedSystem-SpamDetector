import sys
import os
import re
import warnings
from io import open

import pyspark as ps
from pyspark.sql.types import StringType
from pyspark.sql import functions
from pyspark.sql import types

# Run this file will produce 2 csv folders: train_data.csv and test_data.csv which can be used for training the NB Classifier
# Python 2.7 + re + io + pyspark libraries are required

# Load the file from the 
def loadFile(sFilename):
    global directory
    # Given a file name, return the contents of the file as a string.
    f = open(directory + sFilename, "r", encoding = "latin-1")
    sTxt = f.read()
    f.close()
    # Replace all the newline char with white space
    sTxt = sTxt.replace('\n', ' ')
    # Remove all the special characters
    sTxt = re.sub(r'[^A-Za-z0-9 ]','',sTxt)
    return sTxt

# This function goes through each line of RDD, get the filename, load the file content, and add it to content column of RDD
def loadData(dataframe):
    loadDataUDF = functions.udf(lambda x: loadFile(x), types.StringType())
    return dataframe.withColumn('content', loadDataUDF('file'))

if __name__ == "__main__":
    # Create a SparkContext
    try:
        sc = ps.SparkContext.getOrCreate()
        sc.setLogLevel("ERROR")
        sqlContext = ps.sql.SQLContext(sc)
    except ValueError:
        warnings.warn('SparkContext already exists in this scope')

    # Directory of the txt files used for training
    directory = 'train_data/'
    trainFiles = []

    for fFileObj in os.walk(directory):    
        for filename in fFileObj[2]:
            # For each file, save the label and the filename
            trainFiles.append(('ham' if 'ham' in filename else 'spam', filename))
            
    # Create a dataframe for the file list
    df = sc.parallelize(trainFiles).toDF(['label', 'file'])

    # Load the files
    loadedDF = loadData(df)

    # Write to CSV file
    loadedDF.coalesce(1).write.csv("train_data.csv")

    # Directory of the txt files used for testing
    directory = 'test_data/'
    trainFiles = []

    for fFileObj in os.walk(directory):    
        for filename in fFileObj[2]:
            # For each file, save the label and the filename
            trainFiles.append(('ham' if 'ham' in filename else 'spam', filename))
            
    # Create a dataframe for the file list
    df = sc.parallelize(trainFiles).toDF(['label', 'file'])

    # Load the files
    loadedDF = loadData(df)

    # Write to CSV file
    loadedDF.coalesce(1).write.csv("test_data.csv")