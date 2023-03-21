import pandas as pd
import numpy as np

dataset = pd.read_csv('./spam.csv',encoding="ISO-8859-1")

x=dataset['v1']
# print(x)
y=dataset['v2']
# print(y)

x=np.array(x)
y=np.array(y)

text_lowercase=[]
for i in range(len(y)):
  text_lowercase.append(str(y[i]).lower())

print(y[0])
print(text_lowercase[0])

import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

filtered_data=[]
for i in range(len(text_lowercase)):
  text = text_lowercase[i]
  filtered_text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
  filtered_data.append(filtered_text)

# print(text_lowercase[0])
print(filtered_data[0])

from sklearn import preprocessing

y=filtered_data
y_=pd.DataFrame(y)
encoder=preprocessing.LabelEncoder()
y_onehotencoding  = y_.apply(encoder.fit_transform)
# print(y_onehotencoding.head())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

vectorizer_onehot=preprocessing.OneHotEncoder()
vectorizer_onehot.fit(y_onehotencoding)
y_vector_onehot=vectorizer_onehot.transform(y_onehotencoding).toarray()
print(y_vector_onehot)
print("\n")

x_train,x_test,y_train,y_test=train_test_split(y_vector_onehot,x,random_state=1,test_size=0.4)
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)

y_pred=log_reg.predict(x_test)
print(accuracy_score(y_test,y_pred))

