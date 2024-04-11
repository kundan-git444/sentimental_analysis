import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax



Model = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(Model)
model = AutoModelForSequenceClassification.from_pretrained(Model)

plt.style.use('ggplot')




def polarity_scores_roberta(example):
    labels = ['Negative', 'Neutral', 'Positive']
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment = labels[scores.argmax()]
    sentiment = [sentiment, max(scores)]
    return sentiment
print(polarity_scores_roberta("I dont like this")[0])