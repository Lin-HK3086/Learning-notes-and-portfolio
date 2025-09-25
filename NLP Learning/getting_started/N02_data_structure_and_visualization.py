import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud

from N01_reproducibility_and_data_download import *

# Take a first look at the data
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape:     {test_df.shape}")
print(train_df.head())

# Check for missing values
print("Missing values per column:")
print(train_df.isnull().sum())

# Target distribution
print("\nTarget distribution:")
print(train_df['target'].value_counts())

# Download stopwords if needed
stop_words = set(stopwords.words('english'))

def create_wordcloud(text_series, title):
    # Combine all text
    text = ' '.join(text_series)

    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          stopwords=stop_words,
                          max_words=150,
                          collocations=False).generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

# Create word clouds for each class
create_wordcloud(train_df[train_df['target'] == 1]['text'],
                 'Words in Disaster Tweets')
print()
create_wordcloud(train_df[train_df['target'] == 0]['text'],
                 'Words in Non-Disaster Tweets')