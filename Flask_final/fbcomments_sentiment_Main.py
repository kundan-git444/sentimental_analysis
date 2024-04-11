from flask import Flask, render_template, request
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})



from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


app = Flask(__name__)


Model = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(Model)
model = AutoModelForSequenceClassification.from_pretrained(Model)


# ***********************Preprocessing************************************
import pandas as pd
import re

path = "csv files\ESUIT _ Comments Exporter for Facebook™ (200).csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(path)

Number_of_comments = len(df)
#downscale for easy running
df = df.head(10)  #selecting top 10 comments

# Define a function to remove special characters and non-English text
def clean_text(text):
    # Remove links starting with https://
    text = re.sub(r'https://\S+', '', text)
    # Remove non-English characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define the columns you want to keep
selected_columns = ['Author', 'Content', 'CommentAt','ReactionsCount','SubCommentsCount']  # Add the column names you want to keep

# Remove all other columns
df = df[selected_columns]

# Apply the clean_text function to the column containing the text
df['cleaned_text'] = df['Content'].apply(clean_text)
df['Author'] = df['Author'].apply(clean_text)

# Replace empty cells in the 'cleaned_text' column with '-'
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: '-' if pd.isna(x) or x.strip() == '' else x)
df['Author'] = df['Author'].apply(lambda x: 'Unknown_name' if pd.isna(x) or x.strip() == '' else x)
print(df.head())




#*************************************************************************

# Function to get sentiment from text
def polarity_scores_roberta(example):
    labels = ['Negative', 'Neutral', 'Positive']
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment = labels[scores.argmax()]
    sentiment = [sentiment,max(scores)]
    return sentiment




@app.route('/')
def index():
    # Read the cleaned comments from CSV file
    
    #df = pd.read_csv(file, encoding='UTF-8')

    # Process the comments and count sentiments
    negative_count, neutral_count, positive_count = 0, 0, 0


    #comment with most reactions
    reactioncount = 0
    mostreactedcomment = "None"
    
    #most positive comment
    positive_score = 0
    mostpositivecomment = "None"

    #most negative comment
    negative_score = 0
    mostnegativecomment = "None"



    for i, row in df.iterrows():
        text = row['cleaned_text']
        tempcount = row['ReactionsCount']
        if tempcount > reactioncount:
            reactioncount = tempcount
            mostreactedcomment = row['Content']
        temp_list = polarity_scores_roberta(text)[0]
        roberta_result = temp_list[0]
        score = temp_list[1]
        
        if roberta_result == "Negative":
            negative_count += 1
            if score > negative_score:
                negative_score = score 
                mostnegativecomment = row['Content']
            
        elif roberta_result == "Neutral":
            neutral_count += 1
        elif roberta_result == "Positive":
            positive_count += 1
            if score > positive_score:
                positive_score = score 
                mostpositivecomment = row['Content']


    categories =["Negative","Neutral", "Positive"]
    values = [negative_count, neutral_count, positive_count]
    plt.figure(figsize=(10, 6))
    colors = ['#db4437','#f4b400','#0e9d58']
    plt.barh(categories, values, color=colors)
    plt.xlabel('Count')
    plt.ylabel('Sentiment')
    plt.title('Sentiment_Analysis')
    plt.savefig("Flask_final\static\plot.png")

   


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
    
    categories = []
    values = []

    for keyword, freq in most_common_keywords:
        categories.append(keyword)
        values.append(freq)

    
    plt.figure(figsize=(10, 6))
    plt.barh(categories, values, color='#2bc2c2')
    plt.xlabel('Count')
    plt.ylabel('Freqent words')
    plt.title('Most Common Words')
    plt.savefig("Flask_final\static\plot2.png")
    
    return render_template('index.html', 
                           negative_count = negative_count,
                           neutral_count = neutral_count,
                           positive_count = positive_count,
                           mostreactedcomment = mostreactedcomment,
                           reactioncount = reactioncount,
                           selectedfile = path[10:],
                           Number_of_comments = Number_of_comments,
                           mostpositivecomment = mostpositivecomment,
                           positive_score = positive_score,
                           mostnegativecomment = mostnegativecomment,
                           negative_score = negative_score)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        comment = request.form['comment']
        sentiment = polarity_scores_roberta(comment)[0]
        return render_template('analyze.html', sentiment=sentiment, comment=comment)
    return render_template('analyze.html')


if __name__ == '__main__':
    app.run(debug=False)
