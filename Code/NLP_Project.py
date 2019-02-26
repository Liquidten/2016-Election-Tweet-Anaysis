"""
Created on Sun Apr 29 20:14:48 2018

@author: albert
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import metrics
from sklearn import ensemble

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier 

#approximately 2 minutes to run everything

def all_mentions(tw):
    Filter = re.findall('(\@[A-Za-z_]+)', tw)
    if Filter:
        return Filter
    else:
        return ""

def get_hashtags(tw):
    test3 = re.findall('(\#[A-Za-z_]+)', tw)
    if test3:
        return test3
    else:
        return ""
    

def candidate_code(x):
    if x == 'HillaryClinton':
        return 1
    elif x == 'realDonaldTrump':
        return 0
    else:
        return ''
    
def split_into_tokens(message):
    message = message  # convert bytes into proper unicode
    return TextBlob(message).words

def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]
    

df1 = pd.read_csv('tweets.csv', encoding="utf-8")
df1 = df1[['handle','text','is_retweet']]

df = df1.loc[df1['is_retweet'] == False]
df = df.copy().reset_index(drop=True)
df['top_mentions'] = df['text'].apply(lambda x: all_mentions(x))
df['top_hashtags'] = df['text'].apply(lambda x: get_hashtags(x))
df['length_no_url'] = df['text']
df['length_no_url'] = df['length_no_url'].apply(lambda x: len(x.lower().split('http')[0]))
df['message'] = df['text'].apply(lambda x: x.lower().split('http')[0])
df['label'] = df['handle'].apply(lambda x: candidate_code(x))


messages = df[['label','message']]

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
#print((bow_transformer.vocabulary_))


msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.3, random_state=1)

# Creating pipeline for 3 different classifiers
# Pipeline meaning the data will be processed in the following manner:
    # Spliting words to lemmas  --> Applying TFIDF to every woord
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', ensemble.RandomForestClassifier()),  # train on TF-IDF vectors w/ Random Forest classifier
])
    
pipeline2 = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])    
    
pipeline3 = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', DecisionTreeClassifier()),  # train on TF-IDF vectors w/ Decision Tree classifier
])

#Parameter for grid Search    
params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

#Applying Grid Search on 3 different Classifiers
gridRF = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    scoring='accuracy',  # what score are we optimizing?
    cv =5 # what type of cross validation to use
)



gridMNB = GridSearchCV(
    pipeline2,  # pipeline from above
    params,  # parameters to tune via cross validation
    scoring='accuracy',  # what score are we optimizing?
    cv=5 # what type of cross validation to use
)



gridDC = GridSearchCV(
    pipeline3,  # pipeline from above
    params,  # parameters to tune via cross validation
    scoring='accuracy',  # what score are we optimizing?
    cv=5,  # what type of cross validation to use
)


gridRF.fit(msg_train, label_train)
predictions = gridRF.predict(msg_test)
print(classification_report(label_test, predictions))
print("Accuracy on test set:  %.2f%%" % (100 * (gridRF.score(msg_test,label_test))))


gridMNB.fit(msg_train, label_train)
predictions2 = gridMNB.predict(msg_test)
print(classification_report(label_test, predictions2))
print("Accuracy on test set:  %.2f%%" % (100 * gridMNB.score(msg_test,label_test)))


gridDC.fit(msg_train, label_train)
predictions3 = gridDC.predict(msg_test)
print(classification_report(label_test, predictions3))

print("Accuracy on test set:  %.2f%%" % (100 * gridDC.score(msg_test,label_test)))


#Plotting ROC Curve
plt.figure(figsize=(8,8))

fpr, tpr, _ = metrics.roc_curve(label_test,  predictions)
auc = metrics.roc_auc_score(label_test, predictions)
plt.plot(fpr,tpr,color= 'yellowgreen',label="data RFC, auc="+str(auc))
plt.legend(loc=4)

fpr, tpr, _ = metrics.roc_curve(label_test,  predictions2)
auc = metrics.roc_auc_score(label_test,  predictions2)
plt.plot(fpr,tpr,color= 'darkred',label="data MNB, auc="+str(auc))
plt.legend(loc=4)

fpr, tpr, _ = metrics.roc_curve(label_test,  predictions3)
auc = metrics.roc_auc_score(label_test,  predictions3)
plt.plot(fpr,tpr,color= 'lightblue',label="data DT, auc="+str(auc))
plt.legend(loc=4)

lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.title('ROC Plot')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#Plotting Confusion Matrix
fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test, predictions), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
plt.title('Test Set Confusion Matrix RF')

fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test, predictions2), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
plt.title('Test Set Confusion Matrix MNB')

fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test, predictions3), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
plt.title('Test Set Confusion Matrix DT')

plt.show()

#Tweet Test



# ---- this part stays the same----

your_tweet = input("Type done to exit \nENTER YOUR TWEET HERE:  ")

while(your_tweet != 'done'):
    k = (100 * max(gridMNB.predict_proba([your_tweet])[0]))
    i = gridMNB.predict([your_tweet])[0]
    if i == 1:
        i = "Hillary"
    else:
        i = "Trump"
    print("Tweet #1:", "'",your_tweet, "'", ' \n \n', "I'm about %.0f%%" % k,  "sure this was tweeted by", i)
    your_tweet = input("Type done to exit \nENTER YOUR TWEET HERE:  ")