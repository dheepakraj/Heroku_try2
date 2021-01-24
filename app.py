import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# importing required libraries--modified
import numpy as np 
import pandas as pd

from river import feature_extraction
from river import linear_model
from river import metrics
from river import preprocessing
from river import stats

#import text_processing


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def text_processing(dataset, Y=None):
  def count_punct(text):
    try:
        count = sum([1 for char in text if char in string.punctuation])
        return round(count/(len(text) - text.count(" ")), 3)*100
    except:
        return 0

  def count_numr(text):
    try:
        count = sum([1 for char in text.split() if char.isnumeric()])
        return count
    except:
        return 0
  
  dataset['title_len'] = len(dataset['title']) - dataset['title'].count(" ")
  dataset['body_len'] = len(dataset['text']) - dataset['text'].count(" ")
  dataset['punct%'] = count_punct(dataset['title'])

  dataset['title'] = re.sub('[^A-Za-z0-9 ]+', '', dataset['title'])
  dataset['body'] = re.sub('[^A-Za-z0-9 ]+', '', dataset['text'])


  dataset['title_num'] = count_numr(dataset['title'])
  dataset['body_num'] = count_numr(dataset['text'])

  #def cleaning(dataset, Y=None):
  ### Dataset Preprocessing
  #from nltk.corpus import stopwords
  #from nltk.stem.porter import PorterStemmer
  #import re
  ps = PorterStemmer()
  review = re.sub('[^a-zA-Z]', ' ', dataset['title'])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  dataset['title_clean']  =  review


  return dataset


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = {'title':features[1],'text':features[0]}
    final_features=text_processing(final_features)
    prediction = model.predict_one(final_features)

    #output = round(prediction[0], 2)

    return render_template('index.html',prediction_text='The News is {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
