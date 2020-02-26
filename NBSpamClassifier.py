import sys
import re
import warnings
import math
import time

import pyspark as ps
from pyspark.accumulators import AccumulatorParam, AddingAccumulatorParam
from pyspark.sql.types import StructType, StructField, StringType

# Inputs
trainInput = sys.argv[1]
testInput = sys.argv[2]

def tokenize(sText):
    # Given a string of text sText, returns a list of the individual tokens that
    # occur in that string (in order).
    lTokens = []
    sToken = ""
    for c in sText.lower():
        if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
            sToken += c
        else:
            if sToken != "":
                lTokens.append(sToken)
                sToken = ""
            if c.strip() != "":
                lTokens.append(str(c.strip()))

    if sToken != "":
        lTokens.append(sToken)
        
    return lTokens

def getBigrams(unigramList):
    # Get a list of bigrams from the list of non-unique unigrams
    lTokens = []
    for i in range(len(unigramList)-1):
        lTokens.append(unigramList[i] + ' ' + unigramList[i+1])
    return set(lTokens)

def getTrigrams(unigramList):
    # Get a list of trigrams from the list of unigrams
    lTokens = []
    for i in range(len(unigramList)-2):
        lTokens.append(unigramList[i] + ' ' + unigramList[i+1] + ' ' + unigramList[i+2])
    return set(lTokens)

# Accumulator class for dictionary, it's used for sharing data between nodes
class DictAccumulatorParam(AccumulatorParam):
    def zero(self, d):
        return d
    def addInPlace(self, d1, d2):
        for key, value in d2.items():
            if key in d1:
                d1[key] += value
            else:
                d1[key] = value
        return d1

# Accumulator class for dictionary, it's used for sharing data between nodes
class SetAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return initialValue.copy()
    def addInPlace(self, v1, v2):
        return v1.union(v2)

# Train the data
def trainNaiveBayesSpamClassifier(dataframe):
    
    def train(row):
        # Get the category from the row
        category = row.label
        # Get the text from the row
        content = row.content
        # Tokenize the text
        nonUniqueTokens = tokenize(content)
        # Get unique tokens
        tokens = set(nonUniqueTokens)
        # Get unique bigrams
        bigrams = getBigrams(nonUniqueTokens)
        # Get unique trigrams
        trigrams = getTrigrams(nonUniqueTokens)

        # Record all the tokens
        global uniqueTokenAccumulator
        uniqueTokenAccumulator += tokens
        # Record all the bigrams
        global uniqueBigramAccumulator
        uniqueBigramAccumulator += bigrams
        # Record all the trigrams
        global uniqueTrigramAccumulator
        uniqueTrigramAccumulator += trigrams

        # Update counters
        if category == 'ham':
            global hamTrainingAccumulator
            global hamBigramTrainingAccumulator
            global hamTrigramTrainingAccumulator
            hamTrainingAccumulator += dict.fromkeys(tokens, 1)
            hamBigramTrainingAccumulator += dict.fromkeys(bigrams, 1)
            hamTrigramTrainingAccumulator += dict.fromkeys(trigrams, 1)
        else:
            global spamTrainingAccumulator
            global spamBigramTrainingAccumulator
            global spamTrigramTrainingAccumulator
            spamTrainingAccumulator += dict.fromkeys(tokens, 1)
            spamBigramTrainingAccumulator += dict.fromkeys(bigrams, 1)
            spamTrigramTrainingAccumulator += dict.fromkeys(trigrams, 1)
            

        global totalsAccumulator
        totalsAccumulator += { 'all': 1 }
        totalsAccumulator += { category: 1 }

    dataframe.foreach(train)

def predictUsingNaiveBayesSpamClassifier(dataframe):

    def classify(row):
        content = row.content

        # Get the tokens from the content
        nonUniqueTokens = tokenize(content)
        # Get unique tokens
        tokens = set(nonUniqueTokens)
        # Get unique bigrams
        bigrams = getBigrams(nonUniqueTokens)
        # Get unique trigrams
        trigrams = getTrigrams(nonUniqueTokens)

        # Dictionary for the posterior
        posterior = {'ham': 0, 'spam': 0}

        # Get the total
        allCatTotal = float(totals_bc.value['all'])

        for category in {'ham', 'spam'}:

            # This likelihood that this message belongs to this category
            likelihood = 0.
            
            for token in tokens:
                if token in training_bc.value[category]:
                    likelihood += math.log((training_bc.value[category][token] + 1.) /
                        (len(training_bc.value[category]) + uniqueTokenCount_bc.value + 1), 2)
                else:
                    # If the token does not exist in training data, give it a very low probability
                    likelihood += math.log(1. / (len(training_bc.value[category]) +
                                                    uniqueTokenCount_bc.value + 1), 2)
            for bigram in bigrams:
                if bigram in bigram_training_bc.value[category]:
                    likelihood += math.log((bigram_training_bc.value[category][bigram] + 1.) /
                        (len(bigram_training_bc.value[category]) + uniqueBigramCount_bc.value + 1), 2)
                else:
                    # If the bigram does not exist in training data, give it a very low probability
                    likelihood += math.log(1. / (len(bigram_training_bc.value[category]) +
                                                    uniqueBigramCount_bc.value + 1), 2)

            for trigram in trigrams:
                if trigram in trigram_training_bc.value[category]:
                    likelihood += math.log((trigram_training_bc.value[category][trigram] + 1.) /
                        (len(trigram_training_bc.value[category]) + uniqueTrigramCount_bc.value + 1), 2)
                # Assign a very low probability to the term that don't exist in word_counts[cat]
                else:
                    likelihood += math.log(1. / (len(trigram_training_bc.value[category]) +
                                                    uniqueTrigramCount_bc.value + 1), 2)

            # Calculate the posterior
            posterior[category] = math.log(totals_bc.value[category] / allCatTotal, 2) + likelihood
        
        # Save the prediction
        global predictionsAccumulator
        predictionsAccumulator += [(row.label, row.filename, max(posterior.items(), key = lambda x : x[1])[0], posterior['ham'])]
    
    dataframe.foreach(classify)

def displayAccuracyInfo(results):
    cols = zip(*results)

    colActuals = cols[0]
    colPredictions = cols[2]

    # Number of actual spam messages
    numActualSpam = colActuals.count('spam')
    # Number of actual ham messages
    numActualHam = len(colActuals) - numActualSpam

    numCorrectlyIdentifiedSpam = 0
    numCorrectlyIdentifiedHam = 0

    for item in results:
        if item[0] == item[2]:
            if item[0] == 'spam':
                numCorrectlyIdentifiedSpam += 1
            else:
                numCorrectlyIdentifiedHam += 1

    print('Correctly identified spam: ' + str(numCorrectlyIdentifiedSpam) + ' out of ' \
        + str(numActualSpam) + ' or ' + str(numCorrectlyIdentifiedSpam / float(numActualSpam)))
    print('Incorrectly identified spam: ' + str(numActualSpam - numCorrectlyIdentifiedSpam) + ' out of ' \
        + str(numActualSpam) + ' or ' + str((numActualSpam - numCorrectlyIdentifiedSpam) / float(numActualSpam)))
    print('Correctly identified ham: ' + str(numCorrectlyIdentifiedHam) + ' out of ' \
        + str(numActualHam) + ' or ' + str(numCorrectlyIdentifiedHam / float(numActualHam)))
    print('Incorrectly identified ham: ' + str(numActualHam - numCorrectlyIdentifiedHam) + ' out of ' \
        + str(numActualHam) + ' or ' + str((numActualHam - numCorrectlyIdentifiedHam) / float(numActualHam)))
    print('Overall accuracy: ' + str(numCorrectlyIdentifiedSpam + numCorrectlyIdentifiedHam) + ' out of ' \
        + str(len(colActuals)) + ' or ' + str((numCorrectlyIdentifiedSpam + numCorrectlyIdentifiedHam) / float(len(colActuals))))
    

if __name__ == "__main__":
        
    # Create a SparkContext
    try:
        sc = ps.SparkContext.getOrCreate()
        sc.setLogLevel("ERROR")
        sqlContext = ps.sql.SQLContext(sc)
    except ValueError:
        warnings.warn('SparkContext already exists in this scope')

    # Train
    start_time = time.time()

    schema = StructType([
    StructField("label", StringType(), True),
    StructField("filename", StringType(), True),
    StructField("content", StringType(), True)])

    df = sqlContext.read.csv(trainInput,header=False,schema=schema)
    df = df.repartition(40)

    # Train the NaiveBayes model
    uniqueTokenAccumulator = sc.accumulator(set(), SetAccumulatorParam())
    uniqueBigramAccumulator = sc.accumulator(set(), SetAccumulatorParam())
    uniqueTrigramAccumulator = sc.accumulator(set(), SetAccumulatorParam())

    totalsAccumulator = sc.accumulator({}, DictAccumulatorParam())
    totalsAccumulator += {'all': 0}
    totalsAccumulator += {'ham': 0}
    totalsAccumulator += {'spam': 0}

    hamTrainingAccumulator = sc.accumulator({}, DictAccumulatorParam())
    spamTrainingAccumulator = sc.accumulator({}, DictAccumulatorParam())

    hamBigramTrainingAccumulator = sc.accumulator({}, DictAccumulatorParam())
    spamBigramTrainingAccumulator = sc.accumulator({}, DictAccumulatorParam())

    hamTrigramTrainingAccumulator = sc.accumulator({}, DictAccumulatorParam())
    spamTrigramTrainingAccumulator = sc.accumulator({}, DictAccumulatorParam())

    trainNaiveBayesSpamClassifier(df)

    elapsed_time = time.time() - start_time
    print('Total training time: ' + str(elapsed_time))

    ######################################################################

    #Test
    df = sqlContext.read.csv(testInput,header=False,schema=schema)
    df = df.repartition(8)

    start_time = time.time()

    predictionsAccumulator = sc.accumulator([], AddingAccumulatorParam([]))

    totals_bc = sc.broadcast(totalsAccumulator.value)
    training_bc = sc.broadcast({ 'ham': hamTrainingAccumulator.value, 'spam': spamTrainingAccumulator.value })
    bigram_training_bc = sc.broadcast({ 'ham': hamBigramTrainingAccumulator.value, 'spam': spamBigramTrainingAccumulator.value })
    trigram_training_bc = sc.broadcast({ 'ham': hamTrigramTrainingAccumulator.value, 'spam': spamTrigramTrainingAccumulator.value })
    uniqueTokenCount_bc = sc.broadcast(len(uniqueTokenAccumulator.value))
    uniqueBigramCount_bc = sc.broadcast(len(uniqueBigramAccumulator.value))
    uniqueTrigramCount_bc = sc.broadcast(len(uniqueTrigramAccumulator.value))

    predictUsingNaiveBayesSpamClassifier(df)

    elapsed_time = time.time() - start_time
    print('Total classifying time: ' + str(elapsed_time))

    displayAccuracyInfo(predictionsAccumulator.value)

