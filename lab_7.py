from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request
import pickle
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/forward/", methods=['POST'])
def move_forward():
    input_string = request.form['txt']

    fin_feat = np.array(input_string)
    dataset = pd.read_csv('./spam.csv', encoding="ISO-8859-1")

    x = dataset['v1']
    y = dataset['v2']
    x = np.array(x)
    y = np.array(y)

    text_lowercase = []
    for i in range(len(y)):
        text_lowercase.append(str(y[i]).lower())

    stop_words = set(stopwords.words('english'))

    filtered_data = []
    for i in range(len(text_lowercase)):
        text = text_lowercase[i]
        filtered_text = ' '.join(
            [word for word in text.split() if word.lower() not in stop_words])
        filtered_data.append(filtered_text)

    y_new = filtered_data

    tfvect = TfidfVectorizer()
    tfvect.fit(y_new)
    x_tfvect = tfvect.transform(y_new)
    tf_vector = x_tfvect.toarray()
    tf_df = pd.DataFrame(tf_vector)

    pickle.dump(tfvect, open("file2.pkl", "wb"))

    x_train_tf, x_test_tf, y_train_tf, y_test_tf = train_test_split(
        tf_vector, x, test_size=0.15)
    Log = LogisticRegression()
    Log.fit(x_train_tf, y_train_tf)
    y_pred_tf = Log.predict(x_test_tf)
    score = accuracy_score(y_pred_tf, y_test_tf)*100

    pickle.dump(Log, open("file1.pkl", "wb"))

    pickle.dump(score, open("file3.pkl", "wb"))

    tfvect = pickle.load(open('file2.pkl', 'rb'))
    model = pickle.load(open('file1.pkl', 'rb'))
    text = [input_string]
    text = tfvect.transform(text)
    preds = model.predict(text)
    if preds != 'ham':
        return render_template('index.html', value="spam")
    else:
        return render_template('index.html', value="not spam")


if __name__ == '__main__':
    app.run(debug=True)
