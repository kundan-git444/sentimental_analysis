import pandas as pd
import re


path = "C:\\Users\\Kundan\\Desktop\\sentimental_analysis\\csv files\\ESUIT _ Comments Exporter for Facebook™ (200).csv"



# Read the CSV file into a DataFrame
df = pd.read_csv(path)

#downscale for easy running
df = df.head(20)  #selecting top 100 comments

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

# Save the modified DataFrame back to a CSV file
df.to_csv('cleaned_comments.csv', index=False)
