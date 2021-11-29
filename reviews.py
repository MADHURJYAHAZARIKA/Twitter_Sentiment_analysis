# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import StratifiedKFold


#import k_folds_f



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
    
short_pos = open("pos1.txt","r").read()
short_neg = open("neg1.txt","r").read()
short_neg = short_neg.encode('ascii','ignore').decode("utf-8")# to encode

short_pos = short_pos.encode('ascii','ignore').decode("utf-8")
#data cleaning step-3 : removing Emoji --> this can be applied on text

emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
							u"\U0001F300-\U0001F5FF"
							u"\U0001F680-\U0001F6FF"
							u"\U0001F1E0-\U0001F1FF"
							"]+",flags=re.UNICODE)
short_pos = emoji_pattern.sub(r'',short_pos)

short_neg = emoji_pattern.sub(r'',short_neg)
# step-12: Removal of slang
# slang words and their corresponding meaning.
dict_spot = {'ama':'ask me anything','bc':'because','b/c':'because','b4':'because','bae':'before anyone else','bd':'big deal',
	'bf':'boyfriend','bff':'best friends forever','brb':'i will be back soon','btw':'by the way','cu':'see you','cyl':'see you later',
	'dftba':'do not forget to be awesome','dm':'direct message', 'eei5':'explain like i am 5 years old','fb':'facebook','fb':'facebook',
	'fomo':'fear of missing out','ftfy':'fixed this for you','ftw':'for the win','futab':'feet up,take a break','fya':'for your amusement',
	'fye':'for your entertainment','fyi':'for your information','gtg':'got to go','g2g':'got to go','gf':'girlfriend','gr8':'great',
	'gtr':'got to run','hbd':'happy birthday','ht':'hat tip','hth':'here to help','ianad':'i am not a doctor',
	'ianal':'i am not a lawyer','icymi':'in case you missed it','idc':'i dont care','idk':'i dont know','ig':'instagram',
	'iirc':'if i remember correctly','ikr':'i knonw right ?','imo':'in my opinion','imho':'in my honest opinion','irl':'in real life',
	'jk':'just kidding','l8':'late','lmao':'let me know','lol':'laughing out load','mcm':'mam crush monday','myob':'mind your own business',
	'mtfbwy':'may the force be with you','nbd':'no big deal','nm':'not much','nsfw':'not safe for work','nts':'note to self','nvm':'nevermind','oh':'overheard','omg':'oh my god','omw':'on my way','ootd':'outfit of the day','orly':'oh really',
	'pda':'public display of affection','potd':'photo of the day','potus':'president of the united states','pm':'private message','ppl':'people','q':'question','qq':'quick question','qotd':'quote of the day',
	'rofl':'rolling on the floor laughing','roflmao':'rolling on the floor laughing my ass off','rt':'retweet','sfw':'safe for work','sm':'social media','smh':'shaking my head','tbh':'to be honest','tbt':'throwback thursday',
	'tgif':'thank god its friday','thx':'thanks','til':'too much information','tmi':'too much information','ttyi':'talk to you later','ttyn':'talk to you never','ttys':'talk to you soon','txt':'text','w':'with','wbu':'what about you',
	'wcw':'women crush wednesday','wdymbt':'what do you mean by that','wom':'word of mouth','wotd':'word of the day','yolo':'you only live once','yt':'youtube','yw':'youre welcome'}
#striping
short_neg = short_neg.strip()
short_pos = short_pos.strip()





# move this up here
all_words = []
documents = []


#  j is adject, r is adverb, and v is verb
allowed_word_types = ["J","R","V"]
#allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

            #removing slang
for iteri,slang in enumerate(all_words):
    if slang in dict_spot:
        all_words[iteri] = dict_spot[slang]

for iteri,slang in enumerate(all_words):
    if slang in dict_spot:
        all_words[iteri] = dict_spot[slang]

#data cleaning step-2: removing stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
all_words = [w for w in all_words if w not in stop_words]
all_words = [w for w in all_words if w not in stop_words]



save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]


save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
x = [(find_features(rev)) for (rev, category) in documents]
y =  [(category) for (rev, category) in documents]
random.shuffle(featuresets)
skf = StratifiedKFold(n_splits=10)

testing_set = []
training_set =[]

for train_index, test_index in skf.split(x, y):
	for i in train_index:
		training_set.append(featuresets[i])
	for i in test_index:
		testing_set.append(featuresets[i])


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

###############
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)



#SGDC_classifier = SklearnClassifier(SGDClassifier())
#SGDC_classifier.train(training_set)
#print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

#save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
#pickle.dump(SGDC_classifier, save_classifier)
#save_classifier.close()

voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  classifier,
                                  LinearSVC_classifier,
                                  #MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)