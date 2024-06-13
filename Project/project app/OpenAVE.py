# ******************************************
#               Load Libraries
# ******************************************
import streamlit as st
import praw
import pandas as pd
from datetime import datetime, timezone
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt


# ******************************************
#              Configuration
# ******************************************
CLIENT_ID = "2SXthwixaE7lzFEzdE1OPg"
CLIENT_SECRET = "iV-suJoi4ZrTc5B6233M8K0X2mboYw"
USER_AGENT = "Semantic-Analysis-of-a-Social-Media"

# Load BERT model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True)

# Create a Reddit instance to make calls
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Maximum token length
max_token_length = tokenizer.model_max_length

# Function to fetch comments for a given submission
def fetch_comments(submission):
    submission.comments.replace_more(limit=None)
    comments = submission.comments.list()
    data = []
    for comment in comments:
        data.append({
            'comment_id': comment.id,
            'comment_body': comment.body,
            'comment_created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
        })
    return pd.DataFrame(data)


# Perform sentiment analysis on comments
def analyze_sentiment(comment):
    # The pipeline automatically handles tokenization and truncation
    results = sentiment_analyzer(comment)
    # Extract the label with the highest score
    sentiment = results[0]['label']
    return sentiment


# Initialize session state for visibility of analysis and search results
if 'visibility' not in st.session_state:
    st.session_state.visibility = {}

if 'search_results' not in st.session_state:
    st.session_state.search_results = []


# ******************************************
#         Main Streamlit application
# ******************************************
def main():
    # Create tabs
    tabs = st.tabs(["📚 Overview", "🥽 Demo"])

    with tabs[0]:

        # Display the title and image side by side
        col1, col2 = st.columns([3, 1])

        with col1:
            st.title("What is Reddit?")

        with col2:
            st.image("Screenshot.png", width=100)  # Adjust the width as needed

        st.write("## Reddit Overview")

        st.write("""
        - **Social News Aggregation and Discussion Website:**
        - Registered members can submit content, such as text posts, links, images, and videos.
        - Content is organized into various categories called 'subreddits.'
        - Subreddits cover a wide range of topics, from general interests like news, technology, and sports, to niche subjects and communities.
        """)

        st.write("## Subreddits Overview")

        st.write("""
        - **Definition:**
        - Subreddits are individual communities or forums within Reddit that focus on specific topics, interests, or themes.
        - Each subreddit operates like a separate entity under the broader Reddit platform, with its own rules, moderators, and unique culture.
        """)

        st.write("## Key Features of Subreddits:")

        st.write("""
        - **Naming Convention:**
        - Subreddits are named with the prefix "r/" followed by the specific topic (e.g., r/technology, r/cooking, r/funny).
        """)

        st.write("""
        - **Focused Content:**
        - Each subreddit centers around a particular subject or theme. For example:
            - **r/science:** Discussions and news about scientific topics.
            - **r/movies:** Discussions about films, reviews, news, and more.
            - **r/gaming:** Everything related to video games.
            - **r/AskReddit:** A place where users can ask questions and get answers from the community.
        """)

        st.image("Screenshot 2024-06-10 at 23.19.37.png")

        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.text(" ")
        with tabs[0]:
            # Display the title and image side by side
            col1, col2 = st.columns([3, 1])

            with col1:
                st.title("What is BERT? 🤖")

            with col2:
                st.image(
                    "hugging_face_0.png",
                    width=100)  # Adjust the width as needed

        st.write("""
        ## BERT Overview

        - **Bidirectional Encoder Representations from Transformers (BERT):**
          - BERT is a transformer-based model designed for natural language understanding.
          - It was created by Google AI and introduced in a 2018 paper.
          - BERT has significantly advanced the field of NLP by achieving state-of-the-art performance on a wide array of tasks.""")

        st.write("""
        ## Key Features of BERT:

        - **Bidirectional Training:**
          - Unlike traditional models that read text input sequentially (left-to-right or right-to-left), 
          BERT reads the entire sequence of words at once, allowing it to understand the context of a word based on its 
          surrounding words.

        - **Pre-training and Fine-tuning:**
          - BERT is pre-trained on a large corpus of text, such as Wikipedia, using two tasks:
            - **Masked Language Model (MLM):** Randomly masks some tokens in the input and the model must predict the masked tokens.
            - **Next Sentence Prediction (NSP):** Predicts if a given sentence is the subsequent sentence in the original text.
          - After pre-training, BERT can be fine-tuned on specific tasks like question answering, sentiment analysis, and more.
          - In this project, we use a pre-trained BERT model from Hugging Face specifically fine-tuned for sentiment analysis

        - **Transformers Architecture:**
          - BERT is built on the Transformer architecture, which uses self-attention mechanisms to weigh the influence 
          of different words in a sentence.

        - **Applications of BERT:**
          - BERT can be applied to a variety of NLP tasks, such as:
            - **Question Answering:** Answering questions based on a passage of text.
            - **Sentiment Analysis:** Determining the sentiment expressed in a piece of text.
            - **Named Entity Recognition (NER):** Identifying entities like names, dates, and organizations in text.
            - **Text Classification:** Categorizing text into predefined categories.
        """)

        st.text(" ")
        st.text(" ")
        st.image("Screenshot 2024-06-11 at 10.59.56.png", width=600)

    with tabs[1]:
        # Streamlit app
        st.title("Reddit Sentiment Analyzer")

        # User inputs
        subreddit = st.text_input("Enter subreddit (e.g., python):", "stocks")
        search_term = st.text_input("Enter search term:", "NVIDIA")

        if st.button("Search"):
            # Fetch Reddit posts //// Also can limit the number of posts available for the analysis
            search_results = list(reddit.subreddit(subreddit).search(search_term, limit=10))

            if not search_results:
                st.write("No results found.")
            else:
                st.session_state.search_results = search_results
                st.session_state.visibility = {i: False for i in range(len(search_results))}

        # Use session state to display search results and manage visibility
        if st.session_state.search_results:
            st.write("### Found Topics:")

            for i, post in enumerate(st.session_state.search_results):
                post_title = f"{i + 1}. {post.title}"
                st.write(post_title)
                if st.button(f"Toggle Analysis for {i + 1}", key=f"toggle_{i}"):
                    st.session_state.visibility[i] = not st.session_state.visibility[i]

                if st.session_state.visibility.get(i, False):
                    submission = st.session_state.search_results[i]

                    # Fetch comments for selected submission
                    df = fetch_comments(submission)

                    # Perform sentiment analysis on comments
                    df['sentiment'] = df['comment_body'].apply(analyze_sentiment)

                    # Map sentiment labels to mood
                    sentiment_map = {
                        '1 star': 'Bad Mood',
                        '2 stars': 'Bad Mood',
                        '3 stars': 'Neutral Mood',
                        '4 stars': 'Good Mood',
                        '5 stars': 'Good Mood'
                }
                    df['mood'] = df['sentiment'].map(sentiment_map)

                    # Aggregate mood counts over time
                    df['date'] = df['comment_created_utc'].dt.date
                    mood_over_time = df.groupby(['date', 'mood']).size().unstack(fill_value=0)

                    # Display results
                    st.write(f"## Post Title: {submission.title}")
                    st.write(f"Post URL: {submission.url}")
                    st.write(
                        f"Post Date: {datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
                    # st.write("### Mood Analysis:")
                    # st.write(f"Good Mood: {df['mood'].value_counts(normalize=True).get('Good Mood', 0) * 100:.2f}%")
                    # st.write(f"Bad Mood: {df['mood'].value_counts(normalize=True).get('Bad Mood', 0) * 100:.2f}%")
                    # st.write(f"Neutral Mood: {df['mood'].value_counts(normalize=True).get('Neutral Mood', 0) * 100:.2f}%")

                    # Pie chart for sentiment distribution
                    st.write("### Sentiment Distribution:")
                    sentiment_counts = df['mood'].value_counts(normalize=True) * 100
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    plt.title('Sentiment Distribution')
                    st.pyplot(fig)

                    # Line plot for mood over time
                    st.write("### Sentiment Over Time:")
                    st.line_chart(mood_over_time)
                    st.write("### Sample Comments Data:")
                    st.dataframe(df.sample(20))


if __name__ == "__main__":
    main()
