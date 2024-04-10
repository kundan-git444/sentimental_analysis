import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from tqdm.notebook import tqdm

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




plt.style.use('ggplot')

import nltk
import seaborn as sns
import numpy as np


path = "C:\\Users\\Kundan\\Desktop\\sentimental_analysis\\csv files\\ESUIT _ Comments Exporter for Facebook™ (200).csv"

#read data
df = pd.read_csv("cleaned_comments.csv",encoding='UTF-8')

example = df['cleaned_text'][1]
print(example)

#print(nltk.word_tokenize(encoded_example))
Model = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(Model)
model = AutoModelForSequenceClassification.from_pretrained(Model)




def polarity_scores_roberta(example):
    labels = ['Negative', 'Neutral', 'Positive']
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment = labels[scores.argmax()]

    return sentiment

negative_count, neutral_count, positive_count = 0,0,0

m = 0
mostreactedcommment = "None"

for i, row in tqdm(df.iterrows(), total = 20):
    
    
    text = row['cleaned_text']
    Name = row['Author']
    tempcount = row['ReactionsCount']
    if tempcount > m:
        m = tempcount
        mostreactedcommment = row['Content']
        
    roberta_result = polarity_scores_roberta(text)
    #print(Name,',', text, roberta_result)

    if roberta_result == "Negative":
        negative_count += 1
    elif roberta_result == "Neutral":
        neutral_count += 1
    elif roberta_result == "Positive":
        positive_count += 1


print("Negative count: ", negative_count)
print("Neutral count: ", neutral_count)
print("Positive count: ", positive_count)
("Comment with most reactions",m, mostreactedcommment)


    



# Sample comments
comments = df['cleaned_text']

# Combine all comments into a single string
all_text = ' '.join(comments)

# Tokenize the text
tokens = word_tokenize(all_text)

# Remove punctuation
tokens = [word.lower() for word in tokens if word.isalpha()]

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# Count the frequency of each word
word_freq = Counter(tokens)

# Find the most common keywords
most_common_keywords = word_freq.most_common(5) 
print(most_common_keywords)



# Excess

'''
print(df.shape)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # Encode the 'cleaned_text' column to handle non-ASCII characters
    encoded_cleaned_text = df['cleaned_text'].apply(lambda x: x.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
    print(encoded_cleaned_text.to_string())

    
def remove_special_characters(text):
    # Remove special characters using regular expressions
    text = re.sub(r'[^\w\s]', '', text)
    return text


score_dict ={
            "Negative" : scores[0],
            "Neutral" : scores[1],
            "Positive" : scores[2]
        }
'''