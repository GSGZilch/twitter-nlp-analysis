# twitter-nlp-analysis

Script to scrape:
- a given number of tweets
- containing a given hashtag
- starting from a given date

The solution will extract text from the tweets and send to two Azure Cognitive Services endpoints for:
- key phrase extraction
- sentiment analysis

For every given hashtag, a local folder is created in the data directory, containing three subdirectories, one for each sentiment:
- positive
- neutral
- negative

For every sentiment, a CSV file is created containing all key phrases occurring more than given threshold and their respective count, ordered descendingly.

URLs are removed from tweets using regular expressions, to decrease noise in input data.

Optionally, a wordcloud can be displayed for every sentiment, highlighting the most prominent key phrases.