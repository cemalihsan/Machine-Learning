import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, metrics
import wikipedia
import re
import numpy as np

# we can use all page content instead of summary
# wiki_content = wikipedia.page("facebook")
# content_of_wiki = wiki_content.content

content_of_wiki = wikipedia.summary("Artificial_intelligence")

tagged_list = []
abstract_list = []
concrete_list = []

lemmatizer = WordNetLemmatizer()


def get_all_data(filename):  # reads txt file
    with open(filename, mode="r", encoding="utf-8") as text_file:
        data = text_file.read().lower()
    return data


def write_all_data(file, filename):  # write txt file
    with open(filename, mode="w", encoding="utf-8") as text_file:
        text_file.write(file)
    text_file.close()


def read_txt_data(filename, array_list):
    with open(filename, "r") as txt_file:
        for line in txt_file:
            list_array = line.split(" ")
            array_list.append(list_array[0].lower())


def sentence_properties(word, word_index, tagged_words):
    return {
        'word': word,
        'pos': tagged_words[word_index][1],
        'previous-pos': '' if word_index == 0 else tagged_words[word_index - 1][1],
        'two-previous-pos': '' if word_index <= 1 else tagged_words[word_index - 2][1],
        'next-pos': '' if len(tagged_words) - word_index - 1 < 1 else tagged_words[word_index + 1][1],
        'two-next-pos': '' if len(tagged_words) - word_index - 1 < 2 else tagged_words[word_index + 2][1],
        'positive-word': 1 if TextBlob(word).sentiment.polarity > 0 else 0,
        'negative-word': 1 if TextBlob(word).sentiment.polarity < 0 else 0,
        'strong-sub': 1 if TextBlob(word).sentiment.subjectivity > 0.5 else 0,
        'weak-sub': 1 if TextBlob(word).sentiment.subjectivity < 0.5 else 0,
        'around-positive': 1 if TextBlob(tagged_words[word_index - 1][0]).sentiment.polarity > 0 else 1
        if len(tagged_words) - word_index - 1 < 1 else 1 if
        TextBlob(tagged_words[word_index + 1][0]).sentiment.polarity > 0 else 0,
        'around-negative': 1 if TextBlob(tagged_words[word_index - 1][0]).sentiment.polarity < 0 else 1
        if len(tagged_words) - word_index - 1 < 1 else 1 if
        TextBlob(tagged_words[word_index + 1][0]).sentiment.polarity < 0 else 0,
        'abstract-or-concrete': 1 if word in abstract_list else 0,  # if it is 0 then in concrete list
    }

# reads words and its features appended to an array
def read_data_from_sentence(tag_word, tagger):
    for token_index, word in enumerate(tag_word):
        tagged_list.append(sentence_properties(word, token_index, tagger))

# replaces every tag that starts with that word
def replace_with_wn(tag):
    if tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('R'):
        return wn.ADV
    return None

#calculates sentiment polarity
def get_sentiment(text):
    token_count = 0
    negative_count = 0
    word_sent = 0
    plain_texts = nltk.sent_tokenize(text)
    # print(plain_texts)

    for plain_text in plain_texts:
        tagged_text = nltk.pos_tag(nltk.word_tokenize(plain_text))
        # print(tagged_text)

        for word, tag in tagged_text:
            wn_tag = replace_with_wn(tag)

            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
                return []

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                return []

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                return []

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            # print(swn_synset)
            # print("Positive score = " + str(swn_synset.pos_score()))
            # print("Negative score = " + str(swn_synset.neg_score()))
            sentiment = swn_synset.pos_score() - swn_synset.neg_score()
            # print(sentiment)

            if sentiment != 0:
                word_sent += sentiment
                token_count += 1

    if not token_count:
        return 0
    if sentiment >= 0.01:
        return 1
    if sentiment <= -0.01:
        return -1

    return 0


write_all_data(content_of_wiki, "data.txt")
sentence = get_all_data("data.txt")

clean_words = re.sub("[^a-zA-Z]", ' ', sentence)  # eliminates non-alphanumeric characters
sentences = ' '.join(clean_words.split())

stop_words = set(stopwords.words('english'))

tokenized_sent = nltk.sent_tokenize(sentences)
tokenized_word = nltk.word_tokenize(sentences)
# print(tokenized_word)
tagged = nltk.pos_tag(tokenized_word)

filtered_sentence = [w for w in tokenized_word if not w in stop_words]

filtered_sentence = []

for w in tokenized_word:
    if w not in stop_words:
        filtered_sentence.append(w)

read_txt_data("100-400.txt", abstract_list)
read_txt_data("400-700.txt", concrete_list)

read_data_from_sentence(tokenized_word, tagged)  # features of words append to a list

print(tagged_list)

sentimentCalculation = [get_sentiment(x) for x in sentences]

csv_document = pd.DataFrame(tagged_list, columns=['word', 'pos', 'previous-pos', 'two-previous-pos', 'next-pos',
                                                  'two-next-pos', 'positive-word', 'negative-word',
                                                  'strong-sub', 'weak-sub', 'around-positive',
                                                  'around-negative', 'abstract-or-concrete'])

csv_document.to_csv(r'', index=False) #add your path

df = pd.read_csv("features.csv")
df = df.replace(np.nan, '', regex=True)  # filling NaN string values with space

le = preprocessing.LabelEncoder()  # In order convert strings into integers to calculate logistic regression

df['word'] = le.fit_transform(df['word'])
df['pos'] = le.fit_transform(df['pos'])
df['previous-pos'] = le.fit_transform(df['previous-pos'])
df['two-previous-pos'] = le.fit_transform(df['two-previous-pos'])
df['next-pos'] = le.fit_transform(df['next-pos'])
df['two-next-pos'] = le.fit_transform(df['two-next-pos'])
df['positive-word'] = le.fit_transform(df['positive-word'])
df['negative-word'] = le.fit_transform(df['negative-word'])
df['strong-sub'] = le.fit_transform(df['strong-sub'])
df['weak-sub'] = le.fit_transform(df['weak-sub'])
df['around-positive'] = le.fit_transform(df['around-positive'])
df['around-negative'] = le.fit_transform(df['around-negative'])
df['abstract-or-concrete'] = le.fit_transform(df['abstract-or-concrete'])

feature_columns = ['word', 'pos', 'previous-pos', 'two-previous-pos', 'next-pos',
                   'two-next-pos', 'positive-word', 'negative-word',
                   'strong-sub', 'weak-sub', 'around-positive',
                   'around-negative']

x = df[feature_columns]
y = df['abstract-or-concrete']

# print(list(df['abstract-or-concrete']).count(True))
# print(list(df['abstract-or-concrete']).count(False))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LogisticRegression(solver='lbfgs', max_iter=1000000)
model.fit(x_train, y_train)
# print(y_test.size)

# x_x = x_test.to_numpy()
prediction = model.predict(x_test)
# print(prediction.size)

accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)
