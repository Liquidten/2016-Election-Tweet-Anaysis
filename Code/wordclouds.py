import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from wordcloud import STOPWORDS


df = pd.read_csv('tweets.csv')

hills = df.loc[(df['handle'] == 'HillaryClinton'), ['text']]
dons = df.loc[(df['handle'] == 'realDonaldTrump'), ['text']]

stoppers = set(STOPWORDS)
stoppers.add("http")
stoppers.add("https")
stoppers.add("amp")
stoppers.add("CO")
stoppers.add("Trump2016")
stoppers.add("realDonaldTrump")
stoppers.add("will")
stoppers.add("say")
stoppers.add("said")
stoppers.add("let")
stoppers.add("vote")
stoppers.add("now")
stoppers.add("go")
stoppers.add("&amp;")
stoppers.add("i")
stoppers.add("")


    
htops = {}
hillsuni = []
df_hillsuni = pd.DataFrame({'text':[]})
def hills_top_words():
    global htops
    global df_hillsuni
    global hillsuni
    for tweet in hills['text']:
        hillsuni += tweet.split()
    hillsuni = [re.sub("[^a-z@#]+", "", word.lower()) for word in hillsuni]
    df_hillsuni = pd.DataFrame({'text':hillsuni})
    htops = Counter(hillsuni)
    for word in stoppers:
        htops.pop(word, None)
    return htops   
hills_top_words()

hbigs = {}
def hills_top_bigs():
    hillsbigs = []
    global hbigs
    filteredhuni = [word for word in hillsuni if word not in stoppers]
    for i in range(len(filteredhuni)):
        if i < len(filteredhuni)-1:
            hillsbigs.append(filteredhuni[i].lower() +" " + filteredhuni[i+1].lower())    
    hbigs = Counter(hillsbigs)
    return hbigs
hills_top_bigs()                  

def plot_hills_uni():
    labels, values = zip(*htops.most_common(10))
    indexes = np.arange(len(labels))
    width = .5
    plt.barh(indexes, values, width)
    plt.title("Hillary Clinton's Top Unigrams")
    plt.yticks(indexes + width * .0005, labels)
    plt.xlabel("Count")
    plt.savefig('hillaryunigrams.png')
    plt.show()


def plot_hills_big():
    labels, values = zip(*hbigs.most_common(20))
    indexes = np.arange(len(labels))
    width = .5
    plt.barh(indexes, values, width)
    plt.title("Hillary Clinton's Top Bigrams")
    plt.yticks(indexes + width * .0005, labels)
    plt.xlabel("Count")
    plt.savefig('hillarybigrams.png')
    plt.show()

def create_hills_cloud():
    wordcloud_hc = WordCloud(width = 800, height = 400,max_font_size=60, relative_scaling=.1,stopwords=stoppers).generate(hills['text'].str.cat())
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud_hc)
    plt.axis("off")
    plt.savefig('hillarycloud.png', facecolor='k', bbox_inches='tight')
    plt.show()
    
dtops = {}
donsuni = []
df_donsuni = pd.DataFrame({'text':[]})   
def dons_top_words():
    global dtops
    global df_donsuni
    global donsuni
    for tweet in dons['text']:
        donsuni += tweet.split()
    donsuni = [re.sub("[^a-z@#]+", "", word.lower()) for word in donsuni]
    df_donsuni = pd.DataFrame({'text':donsuni})
    dtops = Counter(donsuni)
    for word in stoppers:
        dtops.pop(word, None)
    return dtops     
dons_top_words() 

dbigs = {}
def dons_top_bigs():
    donsbigs = []
    global dbigs
    filteredduni = [word for word in donsuni if word not in stoppers]
    for i in range(len(filteredduni)):
        if i < len(filteredduni)-1:
            donsbigs.append(filteredduni[i].lower() +" " + filteredduni[i+1].lower())    
    dbigs = Counter(donsbigs)
    return dbigs
dons_top_bigs()  

def plot_dons_uni():
    newer = {}
    for key in dtops:
        if dtops[key] > 150:
            newer[key] = dtops[key]
    labels, values = zip(*dtops.most_common(10))
    indexes = np.arange(len(labels))
    width = .5
    plt.barh(indexes, values, width, color='r')
    plt.title("Donald Trump's Top Unigrams")
    plt.xlabel("Count")
    plt.yticks(indexes + width * .0005, labels)
    plt.savefig('donaldunigrams.png')
    plt.show()
    
def plot_dons_big():
    labels, values = zip(*dbigs.most_common(20))
    indexes = np.arange(len(labels))
    width = .5
    plt.barh(indexes, values, width, color='r')
    plt.title("Donald Trump's Top Bigrams")
    plt.xlabel("Count")
    plt.yticks(indexes + width * .0005, labels)
    plt.savefig('donaldbigrams.png')
    plt.show()
    
def create_dons_cloud():
    wordcloud_dt = WordCloud(width = 800, height = 400, max_font_size=60, relative_scaling=.1,stopwords=stoppers).generate(dons['text'].str.cat())
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud_dt)
    plt.axis("off")
    plt.savefig('donaldcloud.png', facecolor='k', bbox_inches='tight')
    plt.show()

plot_hills_uni()
plot_hills_big()
create_hills_cloud()
plot_dons_uni()
plot_dons_big() 
create_dons_cloud()
