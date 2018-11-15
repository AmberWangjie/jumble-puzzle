import json
from itertools import permutations
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf, collect_list
from pyspark.sql.functions import UserDefinedFunction, col
from operator import itemgetter

## Utility function to sort a word
def sortWord(word):
    return ''.join(sorted(word))

## populate freq_dict values from 0->9999
def updateFreqDict():
    global FREQ_DICT
    for k,v in FREQ_DICT.items():
        if v==0:
            FREQ_DICT[k] = MAX_SCORE

## Create dataframe from the input images json (jumbled_images.json)
def createInputDf(input_file):
    print("----- Creating jumbled words df... -----")
    jumbled_json = json.load(open(input_file, "r"))
    jumbled_flat = jumbled_json["inputs"]
    jumbled_fields = [
                        StructField('image_id', LongType(), False),
                        StructField('word', StringType(), False),
                        StructField('circled_spots', ArrayType(IntegerType(),False)),
                        StructField('solution_segments', ArrayType(IntegerType(), False), False)
                    ]
    jumbled_schema = StructType(jumbled_fields)
    jumbled_images_df = spark.createDataFrame(jumbled_flat, jumbled_schema)
    print("----- Jumbled words df created -----")
    return jumbled_images_df

## Find all anagrams of the given "word" in the dictionary
## return a dictionary of anagram:freq
def findAnagramsUDF(word):
    res = {}
    for key,value in FREQ_DICT.items():
        if((len(key) == len(word)) and sortWord(key)==sortWord(word)):
            res[key] = value
    return res

## get circled letters from the anagrams
## return a dictionary of anagram_word: circled_letters 
def getCircledSpotsLetters(anagram_dict, circled_spots):
    circled_dict = {}
    for key,v in anagram_dict.items():
        letter_list = []
        for i in range(len(circled_spots)):
            letter_list.append(key[circled_spots[i]])
        circled_dict[key] = ''.join(letter_list)
    return circled_dict

## aggregate circled letters for puzzles
## return a string containing all circled letters
def aggregateCircledLetters(participants):
    res = ""
    for val in participants:
       for k,v in val.items():
           res+=v
    # print("result is: "+res)
    return res

## given a list of letters, returns all permutations of length wordLen
## letters: string containig the circled letters from the anagrams
## wordLen: length of the words to be created from "letters"
def createAllPerms(letters, wordLen):
    perms = set(''.join(p) for p in permutations(letters, r=wordLen))
    return list(perms)


## create unique and sorted list of strings from the freq_dict
## perms: list of strings
## returns a list of unique permutation words found in the dictionary
def validateFromDict(perms):
    unique_perms = []
    for string in perms:
        if string in FREQ_DICT:
            freq = FREQ_DICT[string]
            if (freq == 0) : freq = 9999
            dictTemp = {}
            dictTemp["key"] = string
            dictTemp["value"] = freq
            unique_perms.append(dictTemp)
    # unique_perms_list = list(unique_perms)
    # res = sorted(unique_perms_list, key=lambda x: FREQ_DICT[x])
    return unique_perms

## remove all letters already used in another word, included in the solution
def removeLetters(letters, word):
    for l in word: 
        letters = letters.replace(l, "", 1)
    return letters

## recursively calculate the all permutations of words of lengths varying
## from segments list values.Stopping condition for the recursion: 
## 1. If the freq count for current set of words reached above frequency threshold paramater
## 2. If we reach the end of segments array
## 3. If there are no more letters left to find permutations from
def recurseFunction(letters,currentList,segments,i,result,currentFreq):
    if(currentFreq >= SCORE_THRESHOLD) : return
    if(i>=len(segments)) :
        result.append({"perm" : '-'.join(currentList), "freq" :str(currentFreq)})
        return
    if(len(letters) <= 0) :
        result.append(currentList)
        return

    perms = createAllPerms(letters, segments[i])
    valid_perms_list = validateFromDict(perms)
    for wordDict in valid_perms_list:
        word = wordDict["key"]
        freq = wordDict["value"]
        newList = list(currentList)
        newList.append(word)
        updated_letters = removeLetters(letters, word)
        recurseFunction(updated_letters,newList,segments,i+1,result,currentFreq+freq)

## Find the final most likely solutions and write to a file   
def finalSolution(segments, letters, image_id):
    print("Finding results for image: ", image_id)
    res = []
    agg_perms_list = []
    recurseFunction(letters,agg_perms_list,segments,0,res,0)
    writeResults(res,image_id)
    return res

## write the results to a file
def writeResults(results,image_id):
    print("Writing results for image: ", image_id)
    results = sorted(results, key=itemgetter('freq'))
    new_list = []
    if len(results) >5:
        for i in range(5):
            new_list.append(results[i])
    else:
        new_list = results
    file_name = "results_better_"+str(image_id)+".txt"
    with open(file_name, "w") as f:
        content = "Solution for image:"+str(image_id)+", is"": "+str(new_list)+"\n"
        f.write(content)
    f.close()

if __name__=="__main__":
    # spark = SparkSession \
    #     .builder \
    #     .appName("Jumbled words puzzle solver") \
    #     .config("spark.executor.memory", "4g") \
    #     .config("spark.num.executors", 2) \
    #     .config("spark.driver.memory", "8g") \
    #     .getOrCreate()
    spark = SparkSession \
        .builder \
        .master("local") \
        .appName("Jumbled words puzzle solver") \
        .getOrCreate()
        
    INPUT_FILE = 'input/jumbled_images-full.json'
    FREQ_FILE = 'input/freq_dict.json'
    FREQ_DICT = json.load(open(FREQ_FILE, "r"))
    MAX_SCORE=9999
    SCORE_THRESHOLD=1100
    # updateFreqDict()

    jumbled_words_df =  createInputDf(INPUT_FILE)
    # jumbled_words_df.show(n=10)
    
    anagram_udf = udf(findAnagramsUDF, MapType(StringType(), IntegerType()))
    jumbled_words_df = jumbled_words_df.withColumn("anagram_dict", anagram_udf(jumbled_words_df.word))
    # jumbled_words_df.show(n=5)
    
    letters_udf = udf(getCircledSpotsLetters, MapType(StringType(), StringType()))
    jumbled_words_df = jumbled_words_df.withColumn("circled_letters_dict", letters_udf(jumbled_words_df.anagram_dict, jumbled_words_df.circled_spots))
    # jumbled_words_df.show(n=5)
    
    aggregate_udf = udf(aggregateCircledLetters, StringType())

    group_df = jumbled_words_df.groupby(['image_id','solution_segments']).agg(collect_list("word").alias("word_list"),
                                                        collect_list("circled_spots").alias("circled_spots_list"),
                                                        collect_list("anagram_dict").alias("anagram_dict_list"),
                                                        aggregate_udf(collect_list("circled_letters_dict")).alias("circled_letters"))
    group_df.show()

    final_udf = udf(finalSolution, ArrayType(MapType(StringType(), StringType())))
    final_df = group_df.withColumn("final_solution",final_udf(group_df.solution_segments, group_df.circled_letters, group_df.image_id))
    final_df.show(n=5)

    spark.stop()