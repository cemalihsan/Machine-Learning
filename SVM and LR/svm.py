import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from nltk.corpus import stopwords

premise_array = []
label_array = []
stop = stopwords.words('english')

df = pd.read_json("multinli_1.0_dev_matched.jsonl", lines=True)
df['sentence1'] = df.sentence1.str.replace(r'[^a-zA-Z ]\s?', r'', regex=True)
df['sentence1'] = df.sentence1.str.lower()
df['sentence2'] = df.sentence1.str.replace(r'[^a-zA-Z ]\s?', r'', regex=True)
df['sentence2'] = df.sentence1.str.lower()
df['premise'] = df['sentence1'] + " " + df['sentence2']

df['premise'] = df['premise'].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in (stop)]))  #removing stop words from sentences
df['premise'] = df['premise'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ') # removing single characters from
# premise

multinli_data = pd.concat([df[c] for c in ["premise", "gold_label"]], axis=1)

multinli_data['gold_label'] = [2 if l == 'contradiction' else 1 if l == 'entailment' else 0 for l in
                               multinli_data.gold_label] #set labels as integers

#print(multinli_data['premise'][0], multinli_data['premise'][1])
#premise_data = multinli_data.to_numpy()
#print(premise_data)

'''
for i in range(len(premise_data)):  #putting 1st and 2nd coloum to different array in order to give as test and train
    for j in range(len(premise_data[0])):
        # print(premise_data[i][0])
        premise_array.append(premise_data[i][0])
        label_array.append(premise_data[i][1])

x = premise_array #premise dataset
y = label_array #gold_label dataset

#print(x)
#print(y)
'''
x = multinli_data['premise'].values
y = multinli_data['gold_label'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2000)

tf_idf = TfidfVectorizer(use_idf=True, analyzer='word')

x_train = tf_idf.fit_transform(X_train)
x_test = tf_idf.transform(X_test)

#Logistic Regression Calculation
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100000)
model.fit(x_train, y_train)
logisticPrediction = model.predict(x_test)
logisticAccuracy = metrics.accuracy_score(y_test, logisticPrediction)
print("Logistic Accuracy:", logisticAccuracy)

#Support Vector Machine Calculation
svmModel = SVC(kernel='linear')
svmModel.fit(x_train, y_train)
svmPrediction = svmModel.predict(x_test)
svmAccuracy = metrics.accuracy_score(y_test, svmPrediction)
print("SVM Accuracy:", svmAccuracy)
