## ABSTRACT
Stock movement prediction using news is a Kaggle Competition to predict how stocks will change based on the market state and
news articles. The aim of this project is to analyze a long series of trading days. For each day, we receive an updated state of the
market, and a series of news articles which were published since the last trading day, along with impacted stocks and sentiment
analysis.This information is then used to predict whether each stock will have increased or decreased ten trading days into
the future. So, ultimately, I have to predict a confidence value between -1 and 1. The target variable to predict here is:
‘returnsOpenNextMktres10’ (marketresidualized return 10 days into the future).IntroductionThe movements in the stock market are 
affected by multiple factors such as the industry’s performance, the company’s performance, the economic status of thecountry etc. 
Similarly news articles also play a major role in influencing the movement of these stock prices. News articles provide information 
regarding the company and the industry’s activities, performance and status, it also provides an estimate or predictions of 
their performance in the future. As investors rely on past and current news while making investments, it can be said that news articles
is one of major factor that influences stocks.


## Dataset
I worked with two datasets: market and news. The training sets of these two datasets have records between 2007 and 2016, while the 
test sets have records between 2017 and 2018. The market training dataset has 4072956 records and 16 features. Market data is 
provided by Intrinio.

Market dataset has the following columns:
- assetCode,assetName: Identifiers for Companies and their different assets
- open/close-stock price at the start andend of that day
- prev/next:returns either looking intopast or future days
- Mktrs/Raw:returns adjusted with somemetrics or raw

The news training dataset has 9328750 records and 35 features. News data is provided by Thomson Reuters. The news dataset has the 
following columns :
- sentimentClass: points to the predominant sentiment class for that particular news item
- sentimentNegative/Positive: probability that sentimient belongs to either class
- noveltyCount:novelty of contents of news items over difference amounts of time (12hrs,3 days and so on)
- volumeCount: aggregate volume of news content belonging to that particular asset over different periods of time(12hrs,3 days and so on)
- provider: the news provider of thenews item
- headlineTag: the headline tag for a particular news item
- marketCommentary: boolean value indicating certain market factors
- Urgency: classifies the news stories urgency (1:alert, 3 :article)
- Relevance:number indicating the relevance of news item to the assets.
The entire list of features can be found on: https://www.kaggle.com/c/two-sigmafinancial-news/data

##Solution

Stocks can either go up or down. So, first, I predict whether the stock will go up or down.So the first part is basically binary
classification. After which I compute the probability of the stock going up through  ‘predict_proba’ function which comes with 
XGBoost (up probability) .Then, I rescale the up probability which is between (0,1) to a new range (-1,1), which is the range of the
target variable. If I know our model confidence for the stock to go up, then our new confidence value is:
Final predicted confidence
value = (2*up probability) - 1
Preprocessing
- I got the useful Looking at the statistics, most data behave homogeneously after 2009 (volume increase, price increase, etc.). But,
before 2009, due to the housing crisis that led to the financial crisis in 2008, the data behaves differently. So the question to make
the right prediction for this problem is: Was there a financial crisis between 2017 and 2018(Testset range)? As, the answer is No,
I only consider records after January 2009.
- I clip records in the market dataset where the difference between open and close price is very high. So, I only consider records
where close price to open price ratio is in the range of 0.5 to 1.5.
- All null values in the market dataset are from market-adjusted columns. So, I fill them up with raw values of the corresponding row.

- Market return should not exceed 50% or fall below 50%. If it does, it is more likely to be noise or extreme data that can confuse our
model. 
- I remove records where asset name is Unknown.
- I clip extreme records in the news Dataset by considering only those falling between the 2nd and 98th quantile.
- Relevance of a news item indicates the significance of the news item to an asset.

News items can be of 2 types: alert or article. I assume that a news item published on a day will be relevant over multiple days based
on the relevance of it. Range of the relevance column is between 0 and 1. If the news about a particular asset appears in the headline,
then the relevance of it is maximum(1). If the news item is of type article, I convert the range of relevance from 0 to 1 to 0 to 7, then
I replicate the news item over the number of days obtained from the new relevance.

While, if the news item is of type alert, I obtain relevance using the formula: Relevance = (sentenceCount -(firstMentionSentence-1))/sentenceCount,
where sentenceCount is the total number of sentences in the news item and firstMentionSentence is the first sentence,
(starting with the headline), in which the scored asset is mentioned. Then, I convert the range of this newly obtained relevance from 
0 to 1 to 0 to 10, and then replicate the news item over the
number of days obtained from the new relevance.
- Irrelevant columns like 'audiences', 'subjects', 'assetName' and 'headline' are removed from the news dataset.
- Categorical columns like 'headlineTag', 'provider' and 'sourceId' are encoded into their numeric representations using the factorize
function. News items having same ‘date’ and‘assetCodes’ are grouped and aggregated on mean.

##Feature engineering
Here, I merge market and news datasets on the columns date and assetCode. I use the function merge to do this and use
only columns from the market dataset, as I need news information for only those records appearing in the market dataset.

 Building Model
- Linear Discriminant Analysis:0.538
- Quadratic Discriminant Analysis:0.512
- AdaBoost(n_estimators=300):0.541
- Logistic Regression (c=1, penalty=l2):0.53
- XGBoost(n_jobs=4,n_estimators=250,max_depth=8,eta=0.1) : 0.62926
- I use XGBoost(eXtreme GradientBoosting) classifier as our final model, as it has the highest accuracy. 
- XGBoost is an implementation of gradient boosting decision tree algorithm designed for speed and performance that is dominating 
competitive machine learning.
- XGBoost was created by Tianqi Chen and its primary benefits are reduced time complexity and better memory utilization.
Gradient boosting is an approach where new models try to reduce the error residuals of the prior models. The new weak learners focus on 
areas where the previous learners performed poorly. After repetitive iterations, the model will be able
to fit the data better. Iterations are done until the error residuals are zero or close to zero. Its aim is to minimize the loss when
adding successive models. It is called gradient boosting because it uses a gradient descent algorithm to minimize theloss when adding 
new models.This approach supports both regression and classification predictive modeling problems.
- I got the highest accuracy of 62.926% with the XGBoost model.
- XGBoost model was chosen for it’s ability to perform incremental learning, a technique where the algorithm steps
through the dataset training on a number of observations without retraining.
- I assume the classifier accuracy is low because there might be a strong correlation between the behavior of stock prices and the time the news
articles become publicly available as [1] suggested. As our classifier model predicts stock returns 10 days into the
future, the dependence of the news article on the stock may have diminished.

## References
- Gidofalvi, Gyozo. Using News Articles to Predict Stock Price Movements. Department
of Computer Science and Engineering, University of California, San Diego. 2001.
- Fung, Gabriel, et. al. The Predicting Power of Textual Information on Financial Markets. IEEE Intelligent Informatics Bulletin.
Vol. 5. No. 1. June 2005.
- Anurag Nagar, Michael Hahsler, Using Text and Data Mining Techniques to extract Stock Market Sentiment from Live News
Streams, IPCSIT vol. XX (2012) IACSIT Press, Singapore.
- Yauheniya Shynkevich, T.M. McGinnity, Sonya Coleman, Ammar Belatreche, Predicting Stock Price Movements Based on Different Categories
of News Articles, 2015  IEEE Symposium Series on Computational Intelligence.
- XGBoost: A Scalable Tree Boosting System -Tianqi Chen,Carlos Guestrin [5]
- Xiqian Zhao, Juan Yang, Lili Zhao & Qing Li, The Impact of News on Stock Market: Quantifying the content of internet based
financial news.
-Data Engineering kernel from public Kernels (Two Sigma Competition)https://www.kaggle.com/dmdm0 2/complete-eda-voting-lightgbm
- Getting Started Kernel from Two Sigma https://www.kaggle.com/dster/two-sigmanews-official-getting-started-kernel
- XGBoost Baseline public Kernel from Kaggle.https://www.kaggle.com/jannesklaas/
lb-0-63-xgboost-baseline.
