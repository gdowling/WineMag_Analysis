import pandas as pd
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import pickle

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

wine_data = pd.read_csv('C:/Users/George/Documents/Kaggle/Wine_Reviews/winemag-data-130k-v2.csv')
top_25 = wine_data.loc[wine_data.points > 90]['description'] #top 25 percent of score for positive 
bottom_25 = wine_data.loc[wine_data.points < 87]['description'] #lower 25 percent for negative

documents = []
all_words = []
selected_tags = ['J','R','V']

for w in top_25:
    documents.append((w,"pos"))
    words = word_tokenize(w)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in selected_tags:
            all_words.append(w[0].lower())
    
for w in bottom_25:
    documents.append((w,"neg"))
    words = word_tokenize(w)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in selected_tags:
            all_words.append(w[0].lower())
            
all_words = nltk.FreqDist(all_words)

#Finding Features
word_features = []
for i in all_words.most_common(6000):
    word_features.append(i[0])

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

training_set = featuresets[:17500]
test_set = featuresets[17500:]

#Fitting Naive Bayes 
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, test_set))*100)
classifier.show_most_informative_features(15)

#Fitting Multinomial Naive Bayes 
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",(nltk.classify.accuracy(MNB_classifier, test_set))*100)

#Fitting Bernoulli Naive Bayes
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",(nltk.classify.accuracy(BNB_classifier, test_set))*100)

#Fitting Logistic Regression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)

#Fitting LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

#Voting    
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set))*100)


def sentiment_classifier_voting(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

def sentiment_score(text):
    feats = find_features(text)
    for prob_pos in LogisticRegression_classifier.prob_classify_many(feats):
        return(prob_pos.prob('pos'))
        print('positive: %.4f negative: %.4f' % (prob_pos.prob('pos'), prob_pos.prob('neg')))

