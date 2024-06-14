# Reddit Sentiment Analyzer

Reddit Sentiment Analyzer is a web application built with Streamlit that performs sentiment analysis on Reddit comments. The application uses the BERT base model from Hugging Face's `transformers` library to analyze the sentiment of comments in various Reddit posts. Users can input a subreddit and a search term, view the found topics, and analyze the sentiment of comments within a selected post.

## Features

- Fetches Reddit posts based on user input (subreddit and search term).
- Analyzes the sentiment of comments in a selected Reddit post using DistilBERT.
- Displays mood analysis percentages (Good, Bad, Neutral).
- Visualizes sentiment distribution with a pie chart.
- Shows the change in mood over time with a line chart.
- Allows toggling of analysis visibility for different posts.

## Limitations

- This application uses the free version of Reddit's API, which has some rate limits and usage restrictions.
- There is a computational constraint that prevents analyzing posts with more than 2,000 comments or it could take a lot of time for a process.


P.S. You will be able to find all the necessary code in the "Project/project app" folder.
