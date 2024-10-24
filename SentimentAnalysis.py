import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
plt.style.use('ggplot')
df =pd.read_csv('/Users/ayushjain/Downloads/Reviews.csv')
df=df.head(500)
print(df.shape)
df.head()
ax=df['Score'].value_counts().sort_index().plot(kind='bar',title='Count of review by stars',figsize=(10,5))
ax.set_xlabel('Review Stars')
plt.show()
example=df['Text'][50]
print(example)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
tokens=nltk.word_tokenize(example)
print(tokens)
nltk.pos_tag(tokens)

#Vader Model

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia=SentimentIntensityAnalyzer()
sia.polarity_scores('I am so happy!')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)
res={}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text=row['Text']
    myId=row['Id']
    res[myId]=sia.polarity_scores(text)
vaders=pd.DataFrame(res).T
vaders=vaders.reset_index().rename(columns={'index':'Id'})
vaders=vaders.merge(df,how='left')
vaders.head()
ax=sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound score by amazon star review')
plt.show()
fig, axs=plt.subplots(1,3,figsize=(14,3))
sns.barplot(data=vaders,x='Score',y='pos',ax=axs[0])
sns.barplot(data=vaders,x='Score',y='neu',ax=axs[1])
sns.barplot(data=vaders,x='Score',y='neg',ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.show()

#roBERTa Model

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
print(example)
sia.polarity_scores(example)
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
results_df.columns
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()
results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['Text'].values[0]
results_df.query('Score == 5').sort_values('vader_neg', ascending=False)['Text'].values[0]

#LIME Implementation

import torch
from transformers import pipeline
import numpy as np
from lime.lime_text import LimeTextExplainer
device = 0 if torch.cuda.is_available() else -1
nlp = pipeline('sentiment-analysis', model='distilbert-base-uncased', device=device)
texts = [
    "I love the service provided by this product!",
    "The customer support was terrible, I am very disappointed.",
    "Amazing experience, will definitely recommend!",
]
def predict_proba(texts):
    results = nlp(texts)
    return np.array([[1 - res['score'], res['score']] for res in results])
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
text_to_explain = "The product quality is amazing, but the delivery was slow."
exp = explainer.explain_instance(text_to_explain, predict_proba, num_features=6)
exp.show_in_notebook(text=True)
text_to_explain = "I love the service provided by this product!"
exp = explainer.explain_instance(text_to_explain, predict_proba, num_features=6)
exp.show_in_notebook(text=True)
