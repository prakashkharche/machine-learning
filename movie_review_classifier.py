import pandas as pd
import nltk.data
from gensim.models import word2vec
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import numpy as np

class DocumentVecorizerModel:
	def __init__(self):
		self.model = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 1000)

	def train(self, reviews):
		self.model.fit_transform(reviews)

	def predict(self, review):
		return self.model.transform([review]).toarray()

	def predict_bulk(self, reviews):
		return self.model.transform(reviews).toarray()

	def save():
		self.model.save()
class ReviewClassifierModel:
	def __init__(self, c):
		self.model = LogisticRegression(C = c)

	def train(self, feature_vectors, training_outputs):
		self.model.fit(feature_vectors, training_outputs)

	def predict(self, review_vector):
		return self.model.predict_proba(review_vector)

class ReviewPredictor:
	def __init__(self):
		pass

	def train(self, data):
		reviews = data["review"]
		cleaned_reviews = [clean_review(review) for review in reviews]
		vectorizer = DocumentVecorizerModel()
		vectorizer.train(cleaned_reviews)
		self.vectorizer_model = vectorizer

		training_data, test_data = split_train_test(data)
		training_feature_vectors = self.vectorizer_model.predict_bulk(training_data["review"])
		training_outputs = training_data["sentiment"].values

		print "Shape of feature_vector %s" % str(training_feature_vectors.shape)
		print "Shape of outputs %s" % str(training_outputs.shape)

		classifier = ReviewClassifierModel(0.2)
		classifier.train(training_feature_vectors, training_outputs)
		self.classifier_model = classifier

		test_feature_vectors = self.vectorizer_model.predict_bulk(test_data["review"])
		test_outputs = test_data["sentiment"].values
		print "Accuracy of classifier is %s" % self.classifier_model.model.score(test_feature_vectors, test_outputs)

	def predict(self, review):
		feature_vector = self.vectorizer_model.predict(review)
		return self.classifier_model.predict(feature_vector)
		
def import_data():
	training_data = pd.read_csv("Bag-popcorn/labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
	return training_data

def convert_sentence_to_words(sentence):
	return sentence.split(" ")

def convert_review_to_sentences(review):
	print "converting review %s" % review
	review_cleaned = ''.join([i if ord(i) < 128 else ' ' for i in review])
	tokenizers = nltk.data.load('tokenizers/punkt/english.pickle')
	raw_sentences = tokenizers.tokenize(review_cleaned.strip())
	sentences = [convert_sentence_to_words(raw_sentence) for raw_sentence in raw_sentences if len(raw_sentence) > 0]
	# for raw_sentence in raw_sentences:
	# 	if len(raw_sentence) > 0:
	# 		words = convert_sentence_to_words(raw_sentence)
	# 		sentences.append(words)
	return sentences
	
def train_word2vec_model():
	training_data = import_data()
	reviews = training_data["review"]
	sentences = []
	for review in reviews:
		sentences += convert_review_to_sentences(review)
	model = word2vec.Word2Vec(sentences, workers = 4, size = 300, min_count = 40, window = 10)
	return model

def clean_review(review):
	# remove non letters
	review_with_letters = re.sub("^a-zA-Z", " ", review)

	words = review_with_letters.split(" ")
	stopwords_list = stopwords.words("english")
	words_cleaned = [w for w in words if not w in stopwords_list]
	return " ".join(words_cleaned)

def train_count_vectorizer_model(data):
	reviews = data["review"]
	cleaned_reviews = [clean_review(review) for review in reviews]
	vectorizer = DocumentVecorizerModel()
	vectorizer.train(cleaned_reviews)

	# training_data, test_data = split_train_test(data)
	# print "Length of training data is %s" % str(len(training_data))
	# print "Length of test data is %s" % str(len(test_data))

	# training_feature_vectors = vectorizer.transform(training_data["review"]).toarray()
	# training_outputs = training_data["sentiment"].toarray()
	return vectorizer

def train_classifier(data, vectorizer, c):
	training_data, test_data = split_train_test(data)
	training_feature_vectors = vectorizer.transform(training_data["review"]).toarray()
	training_outputs = training_data["sentiment"].values
	classifier = ReviewClassifierModel(0.2)
	classifier.train(training_feature_vectors, training_outputs)
	return classifier

def train_models():
	data = import_data()
	review_predictor = ReviewPredictor()
	review_predictor.train(data)
	return review_predictor


def split_train_test(reviews) :
	training_set = reviews[:20000]
	test_data = reviews[20000:]
	return training_set, test_data