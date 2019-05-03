We included a subsample of reviews we found from:  http://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+(Drugs.com)
		We combined the training and testing dataset into a whole dataset and then did our preprocessing analysis.

We included the full FDA Recall dataset we got from: www.fda.gov/Safety/Recalls/default.htm. 

We included our full preprocessing and ML scripts.

We also included our pickle file of our best performing model to test on testing data provided in dataset folder:
		You can use it with following commands in python where drug_reviews_features_data.csv is output of preprocessing:

			import pickle
			import pandas as pd

			with open('submission/best_performing_model_dtree.pickle', 'rb') as handle:
				dtree = pickle.load(handle)

			whole = pd.read_csv("dataset/drugsTestingDataWithFeatures",sep='\t')
			whole = whole.drop(columns = ['Unnamed: 0','drugName', 'condition', 'review', 'rating', 'date'])

			whole = whole.reindex_axis(sorted(whole.columns), axis=1)

			X = whole.drop(columns = ['recall_status']).values
			dtree.predict(X)

			NOTE: The saved model works with the data in dataset folder and not with output of feature_data_preprocessing.py



First, setup pyspark: 
	Ensure Java is installed
	Download spark from http://spark.apache.org/downloads.html (We used 2.3.2 because 2.4 doesn't work on Windows)
	Untar and export export SPARK_HOME and append this to path
	We also used the corresponding version of pyspark as shown in requirements.txt, otherwise there will be a library conflict

Then, install required python libraries:
	pip install -r requirements.txt
	NOTE: We used pip install pyspark==2.3.2, because of the version spark we had, you might have to reinstall pyspark corresponding to the version of spark you have!


Then, setup Stanford NLP.
	Download Stanford NLP: 
		wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
		unzip stanford-corenlp-full-2018-10-05.zip

	In new terminal:
		cd stanford-corenlp-full-2018-10-05
		java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

Then, run preproccessing script in a seperate terminal:
	Ensure proper dependencies are downloaded (we noticed some problems when downloading these within spark submision)
		python -c "import nltk; nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

	Submit pySpark job:
		spark-submit feature_data_preprocessing.py fda_recalls_data.csv drug_reviews_data_subsample.csv

	Run ML model script which will train/test MLP and decision tree:
		python ml.py