## Introduction<br/>
For our project we classified tweets of Donald Trump and Hillary Clinton during the 2016
elections. We created graphs, bigrams, and a prediction method that determines the probability of
the tweet whether it is written by Trump or Hillary.<br/>

## Dataset Description<br/>
We used the dataset: https://www.kaggle.com/benhamner/clinton-trump-tweets<br/>
This dataset provides ~3000 recent tweets and retweets from Hillary Clinton and Donald Trump,
the two major-party presidential nominees during the US Election, from January until September
2016.<br/>
The dataset provides information such as: date, the tweet text, the handle, whether it was a
retweet, amount of likes, amount of retweets, the device the tweet was sent from, and the
location of the tweet. The information used for this project was the text of the tweet, who
tweeted it, and whether or not it was a retweet.<br/>

## Baseline approach description<br/>
<p align="center">
  <img width="460" height="250" src="https://github.com/AlbertSugi/Pictures/blob/master/newdonaldbigram.png"><br/>
fig.1 <br/>
</p>
<p align="center">
  <img width="460" height="250" src="https://github.com/AlbertSugi/Pictures/blob/master/newdonalduni.png"><br/>
fig.2 <br/>
<p align="center">
  <img width="460" height="250" src="https://github.com/AlbertSugi/Pictures/blob/master/newhillbig.png"><br/>
fig.3 <br/>
</p>
</p>
<p align="center">
  <img width="460" height="250" src="https://github.com/AlbertSugi/Pictures/blob/master/newhillunigram.png"><br/>
fig.4 <br/>
</p><br/>

From the dataset we selected the handle, text, is retweet data then using the regress we filtered
out and cleaned each sentences. For preliminary data analysis, we cleaned the sentences using
regional expressions and found the most common unigrams and bigrams for Donald Trump and
Hillary Clinton (fig.1,2,3,4).<br/>
<p align="center">
  <img width="460" height="250" src="https://github.com/AlbertSugi/Pictures/blob/master/donaldcloud.png"><br/>
fig.5 Donald Trump Tweet Analysis WordCloud <br/>
</p>
<p align="center">
  <img width="460" height="250" src="https://github.com/AlbertSugi/Pictures/blob/master/hillarycloud.png"><br/>
fig.6 Hillary Clinton Tweet Analysis WordCloud <br/>
</p><br/>

For this first part, we kept in the hashtags to understand who was
using more hashtags since they are important when writing tweets. We created word clouds for
each candidate since they are useful in conveying and presenting text information (fig.5,6).<br/>

When looking to create a predictor we cleared out the # signs in order to just get the words since
hashtags aren’t that useful when creating a predictor. Then each of the tweet for @HillaryClinton
and @realDonaldTrump were filtered out from the list and given labels are 0 and 1.
Then all of the URLs, hashtags and messages were all split and lowered and combined with their
respective labels. Then a final data frame was created which included labels and messages.
The messages were then converted to tokens and lemmas using TextBlob for further processing.
After that we used the CountVectorizer to convert the messages into a sparse matrix of token
counts and also used the TF-IDF(Term frequency-inverse document frequency) to apply term
frequency inverse document frequency normalization to the sparse matrix we just created.<br/>

## Method description<br/>
We then first created three different pipelines that included 3 different classification models,
which included a Decision Tree, Multinomial Naive Bayes, and an ensemble Random Forest.
Using each of the 3 different pipelines we then ran a grid search using the training data to find
the optimized hyper parameter setting for the dataset. The parameters we tested on were TF-IDF
true or false, and bag of words conversion using lemma or split into tokens. Then this data was
refitted and we used core parallelization which was running on all cores to make the processing
time faster. The scoring optimization was done using accuracy. And finally a stratifiedKFold was
performed using 10 folds.<br/>
<p align="center">
  <img width="460" height="350" src="https://github.com/AlbertSugi/Pictures/blob/master/DTC_CM.png"><br/>
fig.7 Decision Tree Classifier <br/>
</p>
<p align="center">
  <img width="460" height="350" src="https://github.com/AlbertSugi/Pictures/blob/master/MNB_CM.png"><br/>
fig.8 Multinomial Naive Bayes Classifier<br/>
</p>
<p align="center">
  <img width="460" height="350" src="https://github.com/AlbertSugi/Pictures/blob/master/RFC_CM.png"><br/>
fig.9 Random Forrest Classifier<br/>
</p>
Using the grid search parameter setting we performed prediction on the test data. The results
were 84%, 87% and 93% respectively. With MNB scoring the highest accuracy.
Finally a ROC was plotted to show the accuracy comparison of each of the classifier to better
visualize the results. Also a confusion matrix for each classifier was also made to show the fpr,
tpr for each classifier as shown in (fig.7,8,9)<br/>

## Evaluation<br/>
Looking at the results we can see that out of the 3 classifiers we tested on our data set the
Multinomial Naives Bayes outperformed the rest with a test accuracy of 93.61%. While both the
ensemble Random forest and decision tree models were only giving us an accuracy of 87.44%
and 84.57% respectively. <br/>
Inspecting the results, we see the precision scores for our best classifier (MNB) for Hillary
Clinton was 0.94 and for Donald Trump was 0.92. Likewise, the recall scores were 0.93 for each
of the candidates. Therefore, based on the result we can conclude that our classifier is predicting
relevant data more then 90% of the time.<br/>
<p align="center">
  <img width="460" height="400" src="https://github.com/AlbertSugi/Pictures/blob/master/ROC_Plot.png"><br/>
fig.10<br/>
</p>
An ROC plot was plotted to give a visualization of each of the accuracies. Which shows that
curve is inclined towards the true positive values. Meaning good accuracy. (fig.10)<br/>
Lastly, we put an input line where we can test if our input tweet is how much related to each
trump or Hillary. This will predict the users tweet and tell you how much it is accurate to each
candidate.<br/>

## Discussion
This project shows that it is possible to predict who tweeted a certain tweet, based on their past
tweets. By analyzing the language of the past tweets, it is possible to predict whether a tweet is
written by Hillary Clinton or Donald Trump. These individuals have words that they use
frequently, which makes it easier to categorize whether a tweet is from Clinton or Trump. For
example, Trump is more likely to use the words “Make America Great Again,” or “Crooked,”
then Clinton.<br/>
Our results show that the Multinomial Naive Bayes (MNB) is the best at accurately predicting
whether a tweet was written by Donald Trump or Hillary Clinton, with a test accuracy of
93.61%. For analyzing language, this classifier is likely going to perform better than other
classifiers. Looking at the Confusion Matrix (fig.13) the MNB was able to correctly classify 855
of Hillary’s tweets (out of 923) and correctly classify 742 of Donald’s tweets (out of 794). This
outperformed the confusion matrices for the Decision Tree and the Random Forest Classifiers.
As a future application of this work, this predictor could be used to help create a tweet generator.
An individual could choose if they want a tweet from Donald Trump, and the application could
give a randomized Donald Trump tweet based on his most used unigrams and bigrams.
To improve the model accuracy we could also have used neural networks but due to lack in
computing power we were not able to. Also, some of the hyperparameters for the tree models
could have been changed to find the most optimized parameter. Which also we could not
perform due to lack of computer power.<br/>
## Conclusion<br/>
In conclusion, for the twitter dataset we used the best classification model, which was the
Multinomial Naive Bayes with the default hyperparameters. Although the decision tree and
random forest did fairly good, it did not perform as well as the MNB. The reason behind this
could be due to the other two models being tree models whereas MNB is for multinomial
models. So the multinomial classifier models are suitable for classification with discrete features
(example, word counts for test classification). Multinomial models normally require feature
counts. However, in practice fractional counts such as TF-IDF also work which we have used in
our model.
