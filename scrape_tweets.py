import re
import os
import json
import pandas as pd
from pandas import DataFrame
import tweepy
from tweepy.cursor import ItemIterator
from tweepy.models import Status
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Any, Dict, List, Optional, Text, Tuple, Union

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics._models import ExtractKeyPhrasesResult, AnalyzeSentimentResult
from azure.ai.textanalytics._text_analytics_client import TextAnalyticsClient


def dt_print(text: str):
    # helper function to add timestamps to print statements
    print(f"[{datetime.now()}] {text}")


def scrape(company_name: str, 
            date_since: str, 
            num_tweet: int, 
            consumer_key: str, 
            consumer_secret: str, 
            access_key: str, 
            access_secret: str) -> DataFrame:

    # Create empty DataFrame for tweets
    db = pd.DataFrame(columns=['username',
                            'description',
                            'location',
                            'following',
                            'followers',
                            'totaltweets',
                            'retweetcount',
                            'text',
                            'hashtags'])
  
    # OAuth for twitter API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # API call on hashtag
    tweets: ItemIterator = tweepy.Cursor(api.search_tweets,
                        company_name, lang="en",
                        since_id=date_since,
                        tweet_mode='extended').items(num_tweet)
    
    # create list from Tweepy ItemIterator
    list_tweets: List[Status] = [tweet for tweet in tweets]

    # put tweets in DataFrame
    for tweet in list_tweets:
        username: str = tweet.user.screen_name
        description: str = tweet.user.description
        location: str = tweet.user.location
        following: int = tweet.user.friends_count
        followers: int = tweet.user.followers_count
        totaltweets: int = tweet.user.statuses_count
        retweetcount: int = tweet.retweet_count
        hashtags: List[str] = tweet.entities['hashtags']

        try:
            text: str = tweet.retweeted_status.full_text
        except AttributeError:
            text: str = tweet.full_text
        hashtext = list()
        for j in range(len(hashtags)):
            hashtext.append(hashtags[j]['text'])

        ith_tweet: List[Union[str, int]] = [username, description,
                    location, following,
                    followers, totaltweets,
                    retweetcount, text, hashtext]
        db.loc[len(db)] = ith_tweet

    return db


def authenticate_client(key: str, endpoint: str) -> TextAnalyticsClient:
    # authenticate with Azure Cognitive Services client
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    return text_analytics_client


def key_phrase_extraction(client: TextAnalyticsClient, text: str, verbose=False) -> Optional[List[str]]:
    try:
        # API call to Azure Cognitive Services key phrase endpoint
        response: ExtractKeyPhrasesResult = client.extract_key_phrases(documents = text)[0]

        if not response.is_error:
            if verbose:
                # list all key phrases on seperate lines, indented
                dt_print("\tKey Phrases:")
                for phrase in response.key_phrases:
                    dt_print(f"\t\t {phrase}")
            return response.key_phrases
        else:
            if verbose:
                print(response.id, response.error)

    except Exception as err:
        if verbose:
            dt_print("Encountered exception. {}".format(err))

        
def sentiment_analysis(client: TextAnalyticsClient, text: str, verbose=False) -> Tuple[float, float, float]:

    # API call to Azure Cognitive Services sentiment analysis endpoint
    response: AnalyzeSentimentResult = client.analyze_sentiment(documents = text)[0]

    if verbose:
        # list sentiment scores for tweet
        dt_print("\nDocument Sentiment: {}".format(response.sentiment))
        dt_print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
            response.confidence_scores.positive,
            response.confidence_scores.neutral,
            response.confidence_scores.negative,
        ))
    for idx, sentence in enumerate(response.sentences):
        if verbose:
            # list sentiment scores for every sentence in tweet
            dt_print("Sentence: {}".format(sentence.text))
            dt_print("Sentence {} sentiment: {}".format(idx+1, sentence.sentiment))
            dt_print("Sentence score:\nPositive={0:.2f}\nNeutral={1:.2f}\nNegative={2:.2f}\n".format(
                sentence.confidence_scores.positive,
                sentence.confidence_scores.neutral,
                sentence.confidence_scores.negative,
            ))
    return response.confidence_scores.positive, response.confidence_scores.neutral, response.confidence_scores.negative


def calculate(df: DataFrame, client: TextAnalyticsClient) -> List[Dict]:
    # data is a list of observations, see below
    data: List[Dict[str, Union[str, int]]] = []

    for idx, row in enumerate(df.iterrows()):
        # obs(ervation) will hold all data regarding a single tweet, its key phrases and sentiment
        obs: Dict[str, Any] = {}

        # add a unique index for this observation
        obs["index"] = idx
        
        # add username, description, location, ...
        for feat in df.columns.to_list():
            obs[feat.lower()] = row[1][feat]
        
        # extract text from tweet
        tweet_text: str = row[1]["text"]

        # remove URLs from tweets to avoid noisy key phrases
        tweet_text: str = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet_text)
        
        tweet_text_as_lst: List[str] = [tweet_text]
        
        # get keyphrases from Azure Cognitive Services endpoint
        obs["keyphrases"] = key_phrase_extraction(client, tweet_text_as_lst)
        
        # get sentiment from Azure Cognitive Services endpoint
        pos, neutral, neg = sentiment_analysis(client, tweet_text_as_lst)

        # add scores (floats) for every sentiment
        obs["sentiment_scores"] = {
            "positive": pos,
            "neutral": neutral,
            "negative": neg
        }
        
        # add the highest scoring sentiment (str)
        # in case of tie, prioiritize positivity
        max_sentiment_value: float = max([pos, neutral, neg])
        if max_sentiment_value == pos:
            obs["sentiment"] = "positive"
        elif max_sentiment_value == neutral:
            obs["sentiment"] = "neutral"
        else:
            obs["sentiment"] = "negative"

        data.append(obs)

    return data


def get_count_by_sentiment(data: List[Dict], frequency_threshold=2, verbose=False, use_wordcloud=False):
    # list key phrases by sentiment
    kp_by_sentiment: Dict[str, List[str]] = {
        "positive": [],
        "neutral": [],
        "negative": []
    }

    # populate dictionary above
    for obs in data:
        for keyphrase in obs["keyphrases"]:
            # remove company_name from the key phrases
            if keyphrase.lower() != company_name.lower():
                kp_by_sentiment[obs["sentiment"]].append(keyphrase.lower())

    # create dictionary that will hold a DataFrame for every sentiment
    # DateFrame lists number of occurences for every key phrase
    kp_dfs: Dict[str, DataFrame] = {}

    for sentiment in kp_by_sentiment:
        # list of unique key phrases and their count
        kp_counts: List[List[str, int]] = [[keyphrase, kp_by_sentiment[sentiment].count(keyphrase)] for keyphrase in set(kp_by_sentiment[sentiment])]

        # create DataFrame from list of lists
        kp_df = pd.DataFrame(kp_counts, columns=["keyphrase", "count"])

        # get rid of noise by removing low occurence key phrases
        kp_df_cleaned: DataFrame = kp_df[kp_df["count"] >= frequency_threshold]

        # sort by number of occurences
        kp_df_sorted: DataFrame = kp_df_cleaned.sort_values(by="count", ascending=False)

        # add to dictionary
        kp_dfs[sentiment] = kp_df_sorted

        dt_print(f"\{len(kp_df_sorted)} {sentiment} key phrases found with a frequency of at least {frequency_threshold}.")

        # get current time for file paths
        dt = datetime.now()

        # create output path if not exists
        if not os.path.exists(f"{company_name}/{sentiment}_tweets"): os.makedirs(f"{company_name}/{sentiment}_tweets")

        output_filename: str = f"{company_name}/{sentiment}_tweets/{sentiment}_{dt.year}{dt.month:02d}{dt.day:02d}-{dt.hour:02d}{dt.minute:02d}{dt.second:02d}.csv"

        kp_df_sorted.to_csv(output_filename, index=False)

        if verbose:
            print(kp_df_sorted.head(10))

        if use_wordcloud:
            # WordCloud needs all keyphrases in a string seperated by spaces
            kp_str: str = ""
            for kp in kp_by_sentiment[sentiment]:
                kp_str += " " + kp + " "

            # create WordCloud object
            wordcloud = WordCloud(width = 800, height = 800,
                            background_color ='white',
                            min_font_size = 10).generate(kp_str)

            # create and display visual
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.title(f"{sentiment.capitalize()} Topics.")
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
            plt.show()


if __name__ == '__main__':
    # name of the hashtag to filter on
    company_name = "test"

    # get tweets starting from this date, in yyyy-MM-dd format as string
    date_since = "2022-01-01"

    # number of tweets to pull
    num_tweet = 1

    # Twitter API credentials
    consumer_key: str = os.environ["CONSUMER_KEY"]
    consumer_secret: str = os.environ["CONSUMER_SECRET"]
    access_key: str = os.environ["ACCESS_KEY"]
    access_secret: str = os.environ["ACCESS_SECRET"]

    # Azure Cognitive Services key and endpoint
    key: str = os.environ["COG_SERVICES_KEY"]
    endpoint: str = os.environ["COG_SERVICES_ENDPOINT"]

    df_tweets: DataFrame = scrape(company_name, date_since, num_tweet, consumer_key, consumer_secret, access_key, access_secret)

    client: TextAnalyticsClient = authenticate_client(key, endpoint)

    data: List[Dict] = calculate(df_tweets, client)
    
    get_count_by_sentiment(data)