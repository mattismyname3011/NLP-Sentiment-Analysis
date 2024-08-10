# <h1 align='center'> Sentiment Analysis of the Game Genshin Impact Using Naive Bayes </h1>

<div style='display:flex'; gap:5px;>
<div style='font-style:italic; text-align:justify; border:0px solid #4caf50; padding:10px; width:100vw; border-radius:5px;'>
<h3>Abstract</h3>
<p style=''>Indonesia, with a population of 270 million people, has a large market in the gaming industry, which is growing due to economic growth, increased internet access, and higher purchasing power. Genshin Impact, an action RPG game by MiHoYo launched in 2020, has dominated the gaming market and received many reviews from users on Google Play Store, an application download platform. These reviews are crucial for improving the game's quality. This study aims to analyze user sentiment towards the Genshin Impact application using review data from Google Play Store, consisting of both positive and negative sentiments, and employing the Naïve Bayes method. The data analyzed includes 944 Genshin Impact game reviews, with 510 positive and 434 negative. The research results indicate that a 90:10 data split ratio yields a high accuracy rate of 84.2%, a precision rate of 84.48%, and a recall rate of 89.09%, with an average accuracy of 82.9%, an average precision of 82.79%, and an average recall of 87.17%. The Naïve Bayes method has proven effective in providing accurate sentiment analysis results. These analysis results can be used by game developers, particularly MiHoYo, to understand user needs and desires, enhance the Genshin Impact gaming experience, and ultimately increase user satisfaction and loyalty.</p>
</div>

<!-- <div style='font-style:italic; text-align:justify; border:0px solid #4caf50; padding:10px; width:50vw; border-radius:5px;'>
<h3>Abstract</h3>
Indonesia adalah negara dengan jumlah populasi sebanyak 270 juta jiwa, memiliki pasar besar dalam industri game yang semakin berkembang berkat pertumbuhan ekonomi, akses internet yang meningkat, dan daya beli yang lebih tinggi. Genshin Impact merupakan game action RPG dari MiHoYo yang diluncurkan pada tahun 2020 berhasil mendominasi pasar game dan menerima banyak ulasan dari pengguna di Google Play Store yang merupakan platform unduhan aplikasi. Ulasan tersebut penting untuk peningkatan kualitas pada game. Penelitian ini bertujuan menganalisis sentimen pengguna terhadap aplikasi Genshin Impact menggunakan data ulasan pada Google Play Store yang terdiri dari ulasan bersentimen positif dan negatif  serta menggunakan metode Naïve Bayes. Data yang dianalisis terdiri dari 944 ulasan game Genshin Impact dengan kelas positif berjumlah 510 data dan kelas negatif berjumlah 434 data. Hasil penelitian menunjukkan bahwa rasio pembagian data 90:10 menghasilkan nilai akurasi tinggi sebesar 84,2%, presisi sebesar 84.48% dan recall sebesar 89.09% dengan rata-rata akurasi sebesar 82.9%, rata – rata presisi sebesar 82.79% dan rata – rata recall sebesar 87.17%. Metode Naïve Bayes terbukti efektif dalam memberikan hasil analisis sentimen yang akurat. Hasil analisis ini dapat digunakan oleh pengembang game, khususnya MiHoYo, untuk memahami kebutuhan dan keinginan pengguna, serta meningkatkan pengalaman bermain Genshin Impact, yang pada akhirnya dapat meningkatkan kepuasan dan loyalitas pengguna.</p></div> -->

</div>

<!-- table of content -->
<div style='display:flex'; gap:5px;>
<div style="border: 2px solid #4CAF50; padding: 10px; width:30vw; height:100%; border-radius: 5px;">
<h2 style='margin-left:20px;'> Table of Contents </h2>

- [Chapter 1: Data Understanding](#chapter1)
  - [Section 1.1: Import Library](#section_1_1)
  - [Section 1.2: Read Dataset](#section_1_2)
- [Chapter 2: Data Preprocessing](#chapter2)
  - [Section 2.1: Cleansing Function](#section_2_1)
      <!-- * [Sub Section 2.1.1](#sub_section_2_1_1)
      * [Sub Section 2.1.2](#sub_section_2_1_2) -->
  - [Section 2.2: Lemmatizing Function](#section_2_2)
  - [Section 2.3: POS Tagging Function](#section_2_3)
  - [Section 2.4: Preprocessing Process](#section_2_4)
- [Chapter 3: Data Visualization](#chapter3)
- [Chapter 4: Data Preparation](#chapter4)
  - [Section 4.1: Read Dataset](#section_4_1)
  - [Section 4.2: Splitting Dataset](#section_4_2)
  - [Section 4.3: Feature Extraction](#section_4_3)
  - [Section 4.4: Oversample Dataset](#section_4_4)
- [Chapter 5: Modeling](#chapter5)
- [Section 5.1: Multinomial Naive Bayes Model](#section_5_1)
- [Section 5.2: Model Evaluation](#section_5_2)
- [Section 5.3: Prediction Test](#section_5_3)
</div>

<div style="display:flex; flex-direction:column; padding: 10px; width:60vw; height:100%; border-radius: 5px;">

## 1. _Data Understanding_ <a class="anchor" id="chapter1"></a>

#### &ensp;&emsp;_1.1 Import Library_ <a id="section_1_1"></a>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
```

#### &ensp;&emsp;_1.2 Read Dataset_ <a id="section_1_2"></a>

```python
#read csv
df = pd.read_csv('yourfile.csv')
df = df[[
    'content',
    'sentiment'
]]
df['sentiment'].value_counts()

```

## 2. _Data Preprocessing_ <a class="anchor" id="chapter2"></a>

#### &ensp;&emsp;_2.1 Function for Cleaning_ <a id="section_2_1"></a>

```python

#lowercase function
def lowercase(text):
    text = text.lower()
    return text

#abrreviation
def abbreviation(text):
    text=re.sub("isn't",'is not',text)
    text=re.sub("he's",'he is',text)
    text=re.sub("wasn't",'was not',text)
    text=re.sub("there's",'there is',text)
    text=re.sub("couldn't",'could not',text)
    text=re.sub("won't",'will not',text)
    text=re.sub("they're",'they are',text)
    text=re.sub("she's",'she is',text)
    text=re.sub("there's",'there is',text)
    text=re.sub("wouldn't",'would not',text)
    text=re.sub("haven't",'have not',text)
    text=re.sub("that's",'that is',text)
    text=re.sub("you've",'you have',text)
    text=re.sub("te's",'te is',text)
    text=re.sub("what's",'what is',text)
    text=re.sub("weren't",'were not',text)
    text=re.sub("we're",'we are',text)
    text=re.sub("hasn't",'has not',text)
    text=re.sub("you'd",'you would',text)
    text=re.sub("shouldn't",'should not',text)
    text=re.sub("let's",'let us',text)
    text=re.sub("they've",'they have',text)
    text=re.sub("you'll",'you will',text)
    text=re.sub("i'm",'i am',text)
    text=re.sub("we've",'we have',text)
    text=re.sub("it's",'it is',text)
    text=re.sub("don't",'do not',text)
    text=re.sub("that´s",'that is',text)
    text=re.sub("i´m",'i am',text)
    text=re.sub("it’s",'it is',text)
    text=re.sub("she´s",'she is',text)
    text=re.sub("he’s'",'he is',text)
    text=re.sub('i’m','i am',text)
    text=re.sub('i’d','i did',text)
    text=re.sub("he’s'",'he is',text)
    text=re.sub('there’s','there is',text)

    return text

#remove emoticons, url, number, spec.char, puncts
def remove(text):
    text = text.strip(" ")
    text = re.sub('https?://\S+|www\.\S+', '', text) # removing URL links
    text = re.sub(r"\b\d+\b", "", text) # removing number
    text = re.sub('<.*?>+', '', text) # removing special characters,
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # punctuations
    text = re.sub('\n', '', text)
    text = re.sub('[’“”…]', '', text)
    text = re.sub(r'[0-9]', '', text)

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    return text
```

#### &ensp;&emsp;_2.2 Function for Lemmatizing/Stemming_ <a id="section_2_2"></a>

```python
#lemmatize function
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatization_verb(review):
    lemma_result = wordnet_lemmatizer.lemmatize(review, 'v')
    if(lemma_result == review):
        lemma_result = wordnet_lemmatizer.lemmatize(review, 'n')
        if(lemma_result == review):
            lemma_result = wordnet_lemmatizer.lemmatize(review, 'a')
            if(lemma_result == review):
                lemma_result = wordnet_lemmatizer.lemmatize(review, 'r')
                if(lemma_result == review):
                    lemma_result = wordnet_lemmatizer.lemmatize(review, 's')
    return lemma_result
```

#### &ensp;&emsp;_2.3 Function for POS Tagging_ <a id="section_2_3"></a>

```python
# Define a function to perform POS tagging and filter for noun tokens
def pos_tagging(tokens):
    # tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    # Filter tokens that contain 'NN' (noun tags) and return only the words (tokens)
    filtered_tokens = [token[0] for token in tagged if 'NN' in token[1]]
    return filtered_tokens
```

#### &ensp;&emsp;_2.4 Preprocessing Process_ <a id="section_2_4"></a>

```python
#apply function
df['content'] = df['content'].apply(lowercase)
df['content'] = df['content'].apply(abbreviation)
df['content'] = df['content'].apply(remove)
df['content'].head()

#tokenizing
regexp = RegexpTokenizer('\w+')
df['tokenized'] = df['content'].apply(regexp.tokenize)
df['tokenized'].head()

#stopword removal
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
df['tokenized'] = df['tokenized'].apply(lambda x : [item for item in x if item not in stopwords])
df['tokenized'].head()
```

```python
#lemmatizing proses
df['lemmatized'] = df['tokenized'].apply(lambda x: [lemmatization_verb(review) for review in x])
```

```python
#combine string
df['clean_text'] = df['lemmatized'].apply(lambda x : ' '.join([item for item in x]))
df['clean_text']
```

```python
#filter pos tagging to get noun list
df['nouns'] = df['lemmatized'].apply(pos_tagging)
df['nouns']
```

```python
#merge for each list to string and save it
df['nouns_merge'] = df['nouns'].apply(lambda x: ' '.join([item for item in x]))
df.to_csv('./result/dataset_clean.csv', index=False)
```

## 3. _Data Visualization_ <a class="anchor" id="chapter3"></a>

```python
#read clean_dataset
visualize_df = pd.read_csv('./result/dataset_clean.csv')
visualize_df = visualize_df.replace({'POSITIVE': 1, 'NEGATIVE':0})
visualize_df.head()
```

```python
#Working with the most Frequent Words:
from collections import Counter
cnt = Counter()
for text in visualize_df["nouns_merge"].values:
    for word in text.split():
        cnt[word] += 1

# cnt.most_common(10)
temp = pd.DataFrame(cnt.most_common(10))
temp.columns=['word', 'count']
temp
```

```python
FREQWORDS = set([w for (w, wc) in cnt.most_common(1)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])
visualize_df["nouns_merge"] = visualize_df["nouns_merge"].apply(lambda text: remove_freqwords(text))
visualize_df.head()
```

```python
positive_aspect = visualize_df[visualize_df['sentiment'] == 1]
negative_aspect = visualize_df[visualize_df['sentiment'] == 0]
```

```python
#wordcloud for negative nouns
all_negative_aspects = ' '.join(word for word in negative_aspect['nouns_merge'])
wordcloud_negative = WordCloud(colormap='Reds_r', width=500, height=500, mode='RGBA', background_color='black', max_words=100).generate(all_negative_aspects)
plt.figure(figsize=(9,6))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('')
plt.margins(x=0, y=0)
plt.show()
```

```python
#wordcloud for positive nouns
all_positive_aspects = ' '.join(word for word in positive_aspect['nouns_merge'])
wordcloud_positive = WordCloud(colormap='Greens_r', width=500, height=500, mode='RGBA', background_color='black', max_words=100).generate(all_positive_aspects)
plt.figure(figsize=(9,6))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('')
plt.margins(x=0, y=0)
plt.show()
```

## 4. _Data Preparation_ <a class="anchor" id="chapter4"></a>

#### &ensp;&emsp;_4.1 Read Dataset_ <a id="section_4_1"></a>

```python
dataset = pd.read_csv("./result/dataset_clean.csv")
dataset = dataset[[
    'clean_text',
    'sentiment'
]]

dataset = dataset.replace({'NEGATIVE':0, 'POSITIVE':1})
dataset.head()
```

#### &ensp;&emsp;_4.2 Splitting Dataset_ <a id="section_4_2"></a>

```python
X = dataset['clean_text']
y = dataset['sentiment']

X_train73, X_test73, y_train73, y_test73 = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42) #rasio 70:30
X_train82, X_test82, y_train82, y_test82 = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42) #rasio 80:20
X_train91, X_test91, y_train91, y_test91 = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42) #rasio 90:10


ratio_data = {
    'Ratio': ['70:30', '80:20', '90:10'],
    'Train Data' : [len(X_train73),len(X_train82),len(X_train91)],
    'Test Data' : [len(X_test73),len(X_test82),len(X_test91)]
}

ratio_df = pd.DataFrame(ratio_data)
ratio_df
```

#### &ensp;&emsp;_4.3 Feature Extraction_ <a id="section_4_3"></a>

```python
#count vectorizer for training and testing data
count_vectorizer = CountVectorizer()
X_train_count73 = count_vectorizer.fit_transform(X_train73)
X_test_count73  = count_vectorizer.transform(X_test73)

X_train_count82 = count_vectorizer.fit_transform(X_train82)
X_test_count82  = count_vectorizer.transform(X_test82)

X_train_count91 = count_vectorizer.fit_transform(X_train91)
X_test_count91  = count_vectorizer.transform(X_test91)

```

#### &ensp;&emsp;_4.4 Oversample Dataset_ <a id="section_4_4"></a>

```python
smote = SMOTE(random_state=42)
X_train_res73, y_train_res73 = smote.fit_resample(X_train_count73, y_train73)
X_train_res82, y_train_res82 = smote.fit_resample(X_train_count82, y_train82)
X_train_res91, y_train_res91 = smote.fit_resample(X_train_count91, y_train91)
```

## 5. _Modeling_ <a class="anchor" id="chapter5"></a>

#### &ensp;&emsp;_5.1 Multinomial Naive Bayes Model_ <a id="section_5_1"></a>

```python
nb = MultinomialNB()
nb.fit(X_train_res91, y_train_res91)
y_pred = nb.predict(X_test_count91)

```

#### &ensp;&emsp;_5.2 Model Evaluation_ <a id="section_5_2"></a>

```python
#Evaluasi Model
accuracy = accuracy_score(y_test91, y_pred)
classification_rep = classification_report(y_test91, y_pred, target_names=['negative', 'positive'])

print("Naive Bayes Model Accuracy : ", accuracy)
print("\nClassification Report : \n", classification_rep)
```

```python
#confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test91, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['negative', 'positive'])
cmd.plot()
```

#### &ensp;&emsp;_5.3 Prediction Test_ <a id="section_5_3"></a>

```python
import textwrap
text = input("\nEnter a new text to predict: ")
myinput = lowercase(text)
myinput = abbreviation(myinput)
myinput = remove(myinput)
myinput = pd.DataFrame([myinput], columns=['input'])
# myinput = lemmatization_verb(myinput)
myinput['tokenized'] = myinput['input'].apply(regexp.tokenize)
myinput['tokenized'] = myinput['tokenized'].apply(lambda x : [item for item in x if item not in stopwords])
myinput['tokenized'] = myinput['tokenized'].apply(lambda x: [lemmatization_verb(review) for review in x])
myinput['clean_text'] = myinput['tokenized'].apply(lambda x: ' '.join([item for item in x]))
myinput = myinput['clean_text'][0]

myinput_vec = count_vectorizer.transform([myinput])


# myrate = input("\nEnter a rating of the text: ")
predicted_sentiment = nb.predict(myinput_vec)
sentiment_scores = nb.predict_proba(myinput_vec)

if sentiment_scores[0][0] > sentiment_scores[0][1]:
    sentiment_label = 'negative'
else :
    sentiment_label = 'positive'

print(textwrap.fill(text, width=150))
print("="*150)
print(sentiment_label)
```

</div>

</div>
