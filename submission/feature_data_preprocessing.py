from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession

import dateparser

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag

import string
from textblob import TextBlob
from textblob import Word

import sys

from nltk.tokenize import sent_tokenize
import nltk
# nltk.download('vader_lexicon')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pycorenlp import StanfordCoreNLP

# why is this here?
# def parse_condition(string):
#   if np.isnan(string):
#       return ""
#   if "</span>" in string:
#       return ""
#   if "not listed" in string:
#       return ""
#   if len(string) < 4:
#       return ""
#   if '(' in string and (not ')' in string):
#       return string.strip('( ').lower()
#   if (not '(' in string) and ')' in string:
#       return string.strip(') ').lower()
#   return string.lower()

# combined clean_text and lem_stop is this ok? for vader 3 lines commented?
def cleanText(row):    
    verb_exp = ['VB', 'VBZ', 'VBP', 'VBD','VBN','VBG']
    soup = BeautifulSoup(row, 'html.parser')
    #remove code
    for tag in soup.find_all('code'):
        tag.replaceWith(' ')
        
    raw = soup.get_text()
    #remove link
    raw_no_link = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', raw)
    #remove email
    no_link_email = re.sub(r'[\w\.-]+@[\w\.-]+[\.][com|org|ch|uk]{2,3}', "", raw_no_link)
    #remove whitespace
    tab_text = '\t\n\r\x0b\x0c'
    no_link_email_space = "".join([ch for ch in no_link_email if ch not in set(tab_text)])
    #remove fomula
    reg = '(\$.+?\$)|((\\\\begin\{.+?\})(.+?)(\\\\end\{(.+?)\}))'
    raw = re.sub(reg, "", no_link_email_space, flags=re.IGNORECASE)   

    raw = raw.lower()
    #remove numbers
    raw = re.sub('[0-9]+?', ' ', raw) 
    # remove punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    raw = regex.sub(' ', raw)
    #clean out the characters left out after the above step, like we’re, I’m, It’s, i.e.
    raw = re.sub('( s )|( re )|( m )|( i e )',' ',raw) 
    # lementize
    row_t = TextBlob(raw)
    raw = []
    for word, pos in row_t.tags:
        if pos in verb_exp:
            word = Word(word)
            word = word.lemmatize("v")
        else:
            word = Word(word)
            word = word.lemmatize()
        raw.append(word)
    clean = ' '.join(raw)      
    stop_words = set(stopwords.words('english'))
    # remove stop words
    cleaned_text = " ".join([word for word in word_tokenize(clean) if word not in stop_words])  
    return cleaned_text

    # text = BeautifulSoup(raw_text, 'lxml').get_text()  #remove html
    # letters_only = re.sub("[^a-zA-Z]", " ", text)  # remove non-character
    # words = letters_only.lower().split() # convert to lower case 
    # if remove_stopwords: # remove stopword
    #   stops = set(stopwords.words("english"))
    #   words = [w for w in words if not w in stops]
    # if stemming==True: # stemming
    #   # Stemmers remove morphological affixes from words, leaving only the word stem.
    #   # http://www.nltk.org/howto/stem.html
    #   stemmer = SnowballStemmer('english') 
    #   words = [stemmer.stem(w) for w in words]
    # if split_text==True:  # split text
    #   return (words)
    # return(" ".join(words))

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

def TextBlobAnalysis(i):
    i_tr = TextBlob(i)
    return str(i_tr.sentiment[0])+'!'+str(i_tr.sentiment[1])

def VaderAnalysis(inputstring):
    analyser = SentimentIntensityAnalyzer()
    all_sent = sent_tokenize(inputstring)
    totalPositiveCount = 0
    totalNegativeCount = 0
    totalNeutralCount = 0
    totalCompountCount = 0
    totalSentenceCount = 0 
    averagePositiveScore = 0
    averageNegativeScore = 0
    averageNeutralScroe = 0
    averageCompountCount = 0
    for i in range(len(all_sent)):
        sentence = all_sent[i]
        snt = analyser.polarity_scores(sentence)
        totalPositiveCount = totalPositiveCount + snt['pos']
        totalNegativeCount = totalNegativeCount + snt['neg']
        totalNeutralCount = totalNeutralCount + snt['neu']
        totalCompountCount = totalCompountCount + snt['compound']
        totalSentenceCount = totalSentenceCount + 1
    if totalSentenceCount > 0:
        averagePositiveScore = totalPositiveCount/totalSentenceCount
        averageNegativeScore = totalNegativeCount/totalSentenceCount
        averageNeutralScroe = totalNeutralCount/totalSentenceCount
        averageCompountCount = totalCompountCount/totalSentenceCount
    return str(averageNegativeScore)+'!'+str(averageNeutralScroe)+'!'+str(averagePositiveScore)+'!'+str(averageCompountCount)

def StanfordAnalysis(inputstring):
    nlp = StanfordCoreNLP('http://localhost:9000')
    totalPositiveCount = 0
    totalNegativeCount = 0
    totalNeutralCount = 0
    totalSentenceCount = 0 
    averagePositiveScore = 0
    averageNegativeScore = 0
    averageNeutralScroe = 0
    
    res = nlp.annotate(inputstring,
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 10000,
                   })
    try:
        for s in res["sentences"]:
            sentimentValue = int(s["sentimentValue"])
            sentiment = s["sentiment"]
            
            # negative sentiment
            if sentimentValue < 2:
                totalNegativeCount = totalNegativeCount + 1
            elif sentimentValue == 2:
                totalNeutralCount = totalNeutralCount + 1
            else:
                totalPositiveCount = totalPositiveCount + 1
                    
            totalSentenceCount = totalSentenceCount + 1
                
            if totalSentenceCount > 0:
                averagePositiveScore = totalPositiveCount/totalSentenceCount
                averageNegativeScore = totalNegativeCount/totalSentenceCount
                averageNeutralScroe = totalNeutralCount/totalSentenceCount
    except:
        print ("Error encountered!")
    return str(averageNegativeScore)+'!'+str(averageNeutralScroe)+'!'+str(averagePositiveScore)

def derive_basic_features_set(drug_reviews_df, fda_df, spark):
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
    reviews_spark_df = spark.createDataFrame(review_df, reviews_spark_df_schema)
    fda_recalls_spark_df = spark.createDataFrame(fda_df, fda_recalls_spark_df_schema)
    fda_product_descriptions = []
    for row in fda_recalls_spark_df.select("product_description", "recall_date").collect():
        fda_product_descriptions.append([row.product_description, row.recall_date])

    wordCount = udf(lambda s: len(s.split(" ")), IntegerType())
    cleanWords = udf(lambda x: cleanText(x))
    avgWordLength = udf(lambda x: average_word_length(x, is_list=True), FloatType())
    drugRecalledUDF = udf(lambda x: drugRecalled(x, fda_product_descriptions))
    drugRecallDateUDF = udf(lambda x: drugRecallDate(x, fda_product_descriptions))

    textblob_sentimentUDF = udf(lambda x: TextBlobAnalysis(x))
    vader_sentimentUDF = udf(lambda x: VaderAnalysis(x))
    stanford_sentimentUDF = udf(lambda x: StanfordAnalysis(x))

    reviews_spark_df = reviews_spark_df.withColumn("partial_name", f.lower(f.trim(f.split(reviews_spark_df.drug_name, "/")[0]))). \
                                        withColumn("is_recalled", drugRecalledUDF(reviews_spark_df.drug_name)). \
                                        withColumn("review_date", f.to_date(reviews_spark_df.review_date)).fillna('', subset=['review_date']). \
                                        withColumn("condition", f.trim(f.lower(reviews_spark_df.condition))). \
                                        withColumn("review_length", f.length(reviews_spark_df.review)). \
                                        withColumn("review_word_count", wordCount(reviews_spark_df.review)). \
                                        withColumn("cleaned_words", cleanWords(reviews_spark_df.review))

    reviews_added_cols_spark_df = reviews_spark_df.withColumn("review_cleaned_word_count", f.length(reviews_spark_df.cleaned_words)). \
                                                    withColumn("review_avg_word_length", avgWordLength(reviews_spark_df.review)). \
                                                    withColumn("review_avg_cleaned_word_length", avgWordLength(reviews_spark_df.cleaned_words)). \
                                                    withColumn("combined_sentiment", textblob_sentimentUDF(reviews_spark_df.cleaned_words)). \
                                                    withColumn("combined_vader", vader_sentimentUDF(reviews_spark_df.cleaned_words)). \
                                                    withColumn("combined_stanford", stanford_sentimentUDF(reviews_spark_df.cleaned_words))

    split_textblob = f.split(reviews_added_cols_spark_df['combined_sentiment'], '!')
    split_vader = f.split(reviews_added_cols_spark_df['combined_vader'], '!')
    split_stanford = f.split(reviews_added_cols_spark_df['combined_stanford'], '!')

    reviews_spark_df_split = reviews_added_cols_spark_df.withColumn('Sentimental_Polarity', split_textblob.getItem(0)). \
                                                    withColumn('Sentimental_Subjectivity', split_textblob.getItem(1)). \
                                                    withColumn('vader_neg', split_vader.getItem(0)). \
                                                    withColumn('vader_neu', split_vader.getItem(1)). \
                                                    withColumn('vader_pos', split_vader.getItem(2)). \
                                                    withColumn('vader_comp', split_vader.getItem(3)). \
                                                    withColumn('stanford_neg', split_stanford.getItem(0)). \
                                                    withColumn('stanford_neu', split_stanford.getItem(1)). \
                                                    withColumn('stanford_pos', split_stanford.getItem(2))
    return reviews_spark_df_split

fda_recall_input_data_path = sys.argv[1]
drug_review_input_data_path = sys.argv[2]

try:
    fda_df = pd.read_csv(fda_recall_input_data_path, sep=',')
    review_df = pd.read_csv(drug_review_input_data_path, sep=',')
except:
    print("Error loading input files. Please check the path and invoke pipeline again.")
    sys.exit(0)

spark = SparkSession.builder.appName("Feature Data Gen Job").enableHiveSupport().getOrCreate()
english_stopwords = stopwords.words("english")
# spark.sparkContext.broadcast(english_stopwords)

features = derive_basic_features_set(review_df, fda_df, spark)
features.toPandas().to_csv("drug_reviews_features_data.csv")
