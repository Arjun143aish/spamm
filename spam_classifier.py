import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\user\\Documents\\Python\\Heroku-Demo-master\\spam")

messages = pd.read_csv("Spam SMS Collection",sep = '\t', names = ['label','message'])


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
Corpus = []

for i in range(len(messages['message'])):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(words) for words in review if words not in set(stopwords.words('english'))]
    messaging = ' '.join(review)
    Corpus.append(messaging)

    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(Corpus).toarray()
X = pd.DataFrame(X)


Y = pd.get_dummies(messages['label'],drop_first =True)

FullRaw = pd.concat([X,Y], axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw,test_size = 0.3, random_state =123)

Train_X = Train.drop(['spam'], axis =1)
Train_Y = Train['spam'].copy()
Test_X = Test.drop(['spam'], axis =1)
Test_Y  =Test['spam'].copy()

Model = MultinomialNB().fit(Train_X,Train_Y)

Test_pred =Model.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_pred,Test_Y)

sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

import pickle

pickle.dump(Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

pickle.dump(cv,open('cv.pkl','wb'))
CV = pickle.load(open('cv.pkl','rb'))



    

