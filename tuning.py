# This code for tuning
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# load file & Bagging
df =pd.read_csv("googleplaystore_user_reviews.csv",encoding="utf-8")
df = pd.concat([df.Translated_Review, df.Sentiment], axis = 1)
df.dropna(axis = 0, inplace = True)
df.Sentiment = [0 if i=="Positive" else 1 if i== "Negative" else 2 for i in df.Sentiment]

text_list = []

for i in df.Translated_Review :
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    text_list.append(review)



cv = CountVectorizer(max_features = 1000)
x = cv.fit_transform(text_list).toarray()
y = df.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Train data set admit - Random Forest Classifier
classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Chain matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('-Chain Matrix & Accuracy-')
print(cm)
print(accuracy)