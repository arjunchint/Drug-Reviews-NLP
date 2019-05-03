from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.functions import udf

from pyspark import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)



import dateparser

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import lxml
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
import sys


from nltk.corpus import stopwords
english_stopwords = stopwords.words("english")
sc.broadcast(english_stopwords)

def parse_condition(string):
	if np.isnan(string):
		return ""
	if "</span>" in string:
		return ""
	if "not listed" in string:
		return ""
	if len(string) < 4:
		return ""
	if '(' in string and (not ')' in string):
		return string.strip('( ').lower()
	if (not '(' in string) and ')' in string:
		return string.strip(') ').lower()
	return string.lower()

# Convert a raw review to a cleaned review
def cleanText(raw_text, remove_stopwords=True, stemming=False, split_text=True):    
	text = BeautifulSoup(raw_text, 'lxml').get_text()  #remove html
	words = re.sub("[^a-zA-Z]", " ", text).split()  # remove non-character
	# words = letters_only.lower().split() # convert to lower case 
	if remove_stopwords: # remove stopword
		words = list(word.lower() for word in words if word.lower() not in english_stopwords)
	if stemming==True: # stemming
		# Stemmers remove morphological affixes from words, leaving only the word stem.
		# http://www.nltk.org/howto/stem.html
		stemmer = SnowballStemmer('english') 
		words = [stemmer.stem(w) for w in words]
	if split_text==True:  # split text
		return (words)
	return(" ".join(words))

def average_word_length(list_of_words, is_list = True):
	if is_list == True:
		if type(list_of_words) != list:
			list_of_words = list(list_of_words)
		if list_of_words == []:
			return 0
		res = 0
		count = len(list_of_words)
		for item in list_of_words:
			res += len(item)
		return res/float(count)
	list_of_words = list_of_words.split(' ')
	return average_word_length(list_of_words, is_list = True)

def wordCount(self, words_list):
	return np.sum([len(i) for i in words_list])

def drugRecalled(drugName, fdaDrugDescriptions):
	return len([i for i in fdaDrugDescriptions if drugName in i[0]]) > 0

def drugRecallDate(drugName, fdaDrugDescriptions):
	matches = [i for i in fdaDrugDescriptions if drugName in i[0]]
	if len(matches) > 0:
		return matches[0][1]
	else:
		None

def derive_basic_features_set(drug_reviews_df, fda_df):
	reviews_spark_df_schema = StructType([StructField("drug_name", StringType(), False), \
									  StructField("condition", StringType(), True), \
									  StructField("review", StringType(), True), \
									  StructField("rating", IntegerType(), False), \
									  StructField("review_date", StringType(), False), \
									  StructField("useful_count", IntegerType(), False)])
	fda_recalls_spark_df_schema = StructType([StructField("recall_date", StringType(), False), \
									  StructField("brand", StringType(), True), \
									  StructField("product_description", StringType(), True), \
									  StructField("recall_reason", StringType(), False), \
									  StructField("company", StringType(), False)])
	reviews_spark_df = sqlContext.createDataFrame(review_df, reviews_spark_df_schema)
	fda_recalls_spark_df = sqlContext.createDataFrame(fda_df, fda_recalls_spark_df_schema)
	fda_product_descriptions = []
	for row in fda_recalls_spark_df.select("product_description", "recall_date").collect():
		fda_product_descriptions.append([row.product_description, row.recall_date])
	wordCount = udf(lambda s: len(s.split(" ")), IntegerType())
	cleanWords = udf(lambda x: cleanText(x), StringType())
	avgWordLength = udf(lambda x: average_word_length(x, is_list=True), FloatType())
	drugRecalledUDF = udf(lambda x: drugRecalled(x, fda_product_descriptions))
	drugRecallDateUDF = udf(lambda x: drugRecallDate(x, fda_product_descriptions))
	reviews_spark_df = reviews_spark_df.withColumn("partial_name", f.lower(f.trim(f.split(reviews_spark_df.drug_name, "/")[0]))). \
										withColumn("is_recalled", drugRecalledUDF(reviews_spark_df.drug_name)). \
										withColumn("review_date", f.to_date(reviews_spark_df.review_date)).fillna('', subset=['review_date']). \
										withColumn("condition", f.trim(f.lower(reviews_spark_df.condition))). \
										withColumn("review_length", f.length(reviews_spark_df.review)). \
										withColumn("review_word_count", wordCount(reviews_spark_df.review)). \
										withColumn("cleaned_words", cleanWords(reviews_spark_df.review))
	reviews_added_cols_spark_df = reviews_spark_df.withColumn("review_cleaned_word_count", f.length(reviews_spark_df.cleaned_words)). \
													withColumn("review_avg_word_length", avgWordLength(reviews_spark_df.review)). \
													withColumn("review_avg_cleaned_word_length", avgWordLength(reviews_spark_df.cleaned_words))
	return reviews_added_cols_spark_df

fda_recall_input_data_path = sys.argv[1]
drug_review_input_data_path = sys.argv[2]

try:
	fda_df = pd.read_csv(fda_recall_input_data_path, sep=',')
	review_df = pd.read_csv(drug_review_input_data_path, sep=',')
except:
	print("Error loading input files. Please check the path and invoke pipeline again.")
	sys.exit(0)

features = derive_basic_features_set(review_df, fda_df)
features.toPandas().to_csv("drug_reviews_features_data.csv")