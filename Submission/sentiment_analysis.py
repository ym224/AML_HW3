import numpy as np
import string
import re
from sklearn import preprocessing as pp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tokenize import generate_tokens
from matplotlib import pyplot as plt

yelp_file = "yelp_labelled.txt"
amazon_file = "amazon_cells_labelled.txt"
imdb_labelled = "imdb_labelled.txt"
# initialize variables
data, labels = [], []

# parse data set 
def parseDataSet(input_file):
	file = open(input_file, 'r')
	for line in file:
		l = line.strip().split("\t")
		data.append(l[0])
		labels.append(l[1])


def AreLabelsBalanced():
	negative_count = np.unique(labels, return_counts=True)[1][0]
	positive_count = np.unique(labels, return_counts=True)[1][1]
	print (negative_count, positive_count)
	return (negative_count == positive_count)


def preprocess():
	lmt = WordNetLemmatizer()
	# convert all strings to lowercase
	_data = np.char.lower(data)

	# preprocess each review
	for i in range(len(_data)):
		# remove all punctuations and replace apostrophe with space 
		_data[i] = _data[i].replace("'", ' ')
		_data[i] = re.sub('[^\w\s]','',_data[i])
		words = _data[i].split()
		_words = [] # final word list
		for index, word in enumerate(words):
			# remove all stopwords
			if word not in stopwords.words('english'):
				# if not stopword, lemmatize by noun, verb, or adjective
			 	if (lmt.lemmatize(word) != word):
			 		_words.append(lmt.lemmatize(word))
			 	elif (lmt.lemmatize(word, 'v') != word):
			 		_words.append(lmt.lemmatize(word, 'v'))
			 	else:
			 		_words.append(lmt.lemmatize(word, 'a'))
		_data[i] = ' '.join(_words)
	return _data


def splitTrainingAndTesting():
	train_data, test_data, train_labels, test_labels = [], [], [], []
	for i in range(3):
		neg_count = 0
		pos_count = 0
		for j in range(1000*i, 1000*i+1000):
			if labels[j] == '0':
				neg_count += 1	
				if neg_count <= 400:
					train_data.append(data[j])
					train_labels.append(0)
				else:
					test_data.append(data[j])
					test_labels.append(0)
			else:
				pos_count += 1	
				if pos_count <= 400:
					train_data.append(data[j])
					train_labels.append(1)
				else:
					test_data.append(data[j])
					test_labels.append(1)
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)
	return train_data, train_labels, test_data, test_labels


def buildBagOfWords():
	train_feature, test_feature = [], []
	uniqueWords = []
	#build a list of unique words from train data
	for sentence in train_data:
		words = sentence.split()
		for word in words:
			if word not in uniqueWords:
				uniqueWords.append(word)

	# for each review, create a feature vector with ith index as the count of ith word in the wordlist
	for sentence in train_data:
		feat = [0] * len(uniqueWords) 
		for word in sentence.split():	
			feat[uniqueWords.index(word)] += 1
		train_feature.append(feat)

	for sentence in test_data:
		feat = [0] * len(uniqueWords) 	
		for word in sentence.split():
			if word in uniqueWords:
				feat[uniqueWords.index(word)] += 1
		test_feature.append(feat)

	train_feature = np.array(train_feature)
	test_feature = np.array(test_feature)
	return uniqueWords, train_feature, test_feature


def postProcess():
	train_feature_post = pp.normalize(train_feature, norm='l2')
	test_feature_post = pp.normalize(test_feature, norm='l2')
	return train_feature_post, test_feature_post

def trainLogisticRegression(uniqueWords, title):
    # Compute Log Regression
    model = LogisticRegression()
    model.fit(train_feature, train_labels)
    accuracy_score = model.score(test_feature, test_labels)

    print (accuracy_score)

    prediction = model.predict(test_feature)
    cm = confusion_matrix(test_labels, prediction)
    print(cm)
    plotConfusionMatrix(cm, title + '_confusion_matrix_log_reg.png', 'Logistic Regression ')
    	
    return showSignificantWords(model, uniqueWords)

def plotConfusionMatrix(cm, fig_name, title):
	plt.figure()
	plt.imshow(cm, cmap=plt.cm.Blues)
	plt.title(title + 'Confusion Matrix')
	tick_marks = np.arange(2)
	plt.xticks(tick_marks, labels, rotation=45)
	plt.yticks(tick_marks, labels)

	thresh = cm.max() / 2
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(fig_name)

def showSignificantWords(model, uniqueWords):
	# get indices of feature vectors with most weights
    indices = np.argsort(np.absolute(model.coef_))[0]
    sig_words = []
    for i in indices[::-1][:10]:
        sig_words.append(uniqueWords[i])
    print (sig_words)
    return sig_words

def trainNaiveBayesClassifier(uniqueWords, title):
	model = MultinomialNB()
	_train_feature = train_feature.clip(0)
	model.fit(_train_feature, train_labels)
	accuracy_score = model.score(test_feature, test_labels)
	print(accuracy_score)

	prediction = model.predict(test_feature)
	cm = confusion_matrix(test_labels, prediction)
	print(cm)
	plotConfusionMatrix(cm, title + '_confusion_matrix_NB.png', 'Naive Bayes ')

	return showSignificantWords(model, uniqueWords)

def buildNGram(n):
	uniqueWords = set()
	for sentence in train_data:
		words = sentence.split()
		if len(words) <= n:
			uniqueWords.add(sentence)
			continue 
		for i in range(len(words) - n + 1):
			uniqueWords.add(' '.join(words[i:i+n]))
	uniqueWords = list(uniqueWords)

	train_feature, test_feature = [], []
	for sentence in train_data:
		feat = [0] * len(uniqueWords)
		words = sentence.split()
		if len(words) <= n:
			feat[uniqueWords.index(sentence)] += 1
			train_feature.append(feat)
			continue
		for i in range(len(words) - n + 1):
			ngram = ' '.join(words[i:i+n])
			feat[uniqueWords.index(ngram)] += 1
		train_feature.append(feat)

	for sentence in test_data:
		feat = [0] * len(uniqueWords)
		words = sentence.split()
		if len(words) <= n:
			if sentence in uniqueWords:
				feat[uniqueWords.index(sentence)] += 1
				test_feature.append(feat)
				continue
		for i in range(len(words) - n + 1):
			ngram = ' '.join(words[i:i+n])
			if ngram in uniqueWords:
				feat[uniqueWords.index(ngram)] += 1
		test_feature.append(feat) 

	train_feature = np.array(train_feature)
	test_feature = np.array(test_feature)
	return uniqueWords, train_feature, test_feature

# perform pca on processed train and test data 
def performPCA(features, dim):
	feature_means = np.mean(features, axis=0)
	features_adj = features - feature_means
	#_features_adj = features_adj.clip(0)

	V = np.linalg.svd(features_adj)[-1]
	features = features_adj.dot(V[:dim, :].T)
	return features

# parse all three data sets and generate np arrays
parseDataSet(yelp_file)
parseDataSet(amazon_file)
parseDataSet(imdb_labelled)
data = np.array(data)
labels = np.array(labels)

# verify if postive and negative label counts are equal
AreLabelsBalanced()

# preprocess data to remove noise
data = preprocess()

# split data into train and test data
train_data, train_labels, test_data, test_labels = splitTrainingAndTesting()

# extract training and testing features using bag of words method
uniqueWords, train_feature, test_feature = buildBagOfWords()

# apply l2 normalization to train and test features to minimize variance in features
train_feature, test_feature = postProcess()

# build training models, fit testing data and report most significant words based on weighted feature vectors
sig_words = trainLogisticRegression(uniqueWords, 'BoW')
sig_words_NB = trainNaiveBayesClassifier(uniqueWords, 'BoW')

# perform PCA on features from bag of words, with dimension as 10, 50 and 100
# then train logistic regression and naive bayes and report significant words
train_feature = performPCA(train_feature, 10)
test_feature = performPCA(test_feature, 10)
sig_words = trainLogisticRegression(uniqueWords, 'PCA_10_BoW')
sig_words_NB = trainNaiveBayesClassifier(uniqueWords, 'PCA_10_BoW')

# pca with dim 50
train_feature = performPCA(train_feature, 50)
test_feature = performPCA(test_feature, 50)
sig_words = trainLogisticRegression(uniqueWords, 'PCA_50_BoW')
sig_words_NB = trainNaiveBayesClassifier(uniqueWords, 'PCA_50_BoW')

# pca with dim 100
train_feature = performPCA(train_feature, 100)
test_feature = performPCA(test_feature, 100)
sig_words = trainLogisticRegression(uniqueWords, 'PCA_100_BoW')
sig_words_NB = trainNaiveBayesClassifier(uniqueWords, 'PCA_100_BoW')


# build 2-grams and train logistic regression and naive bayes on train and test data
uniqueWords, train_feature, test_feature = buildNGram(2)
sig_words = trainLogisticRegression(uniqueWords, 'Ngram')
sig_words_NB = trainNaiveBayesClassifier(uniqueWords, 'Ngram')

# perform PCA on features from n-gram, with dimension as 10, 50 and 100
# then train logistic regression and naive bayes and report significant words
train_feature = performPCA(train_feature, 10)
test_feature = performPCA(test_feature, 10)
sig_words = trainLogisticRegression(uniqueWords, 'PCA_10_Ngram')
sig_words_NB = trainNaiveBayesClassifier(uniqueWords, 'PCA_10_Ngram')

# pca with dim 50
train_feature = performPCA(train_feature, 50)
test_feature = performPCA(test_feature, 50)
sig_words = trainLogisticRegression(uniqueWords, 'PCA_50_Ngram')
sig_words_NB = trainNaiveBayesClassifier(uniqueWords, 'PCA_50_Ngram')

# pca with dim 100
train_feature = performPCA(train_feature, 100)
test_feature = performPCA(test_feature, 100)
sig_words = trainLogisticRegression(uniqueWords, 'PCA_100_Ngram')
sig_words_NB = trainNaiveBayesClassifier(uniqueWords, 'PCA_100_Ngram')





