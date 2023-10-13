from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
import json
import os

def datasets(path):
    df = pd.read_csv(path)
    X_text = df['text'].values
    X_title = df['title'].values
    y = df['label'].values
    return X_text[:1000], X_title[:1000], y[:1000]



def checking_Unbiasity(y):
    plt.figure(figsize=(7,6))
    labels = 'Fake', 'Real'
    a=0
    b=0
    for i in y:
      if i==0:
        a+=1
      else:
        b+=1
    sizes = [b,a]
    colors = ['lightcoral', 'teal']
    explode = (0.1, 0) 
    plt.rcParams['font.size'] = 18
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()
    plt.close()


def simplify(df,y):
  corpus = []
  for i in range(0, y.size):
    review = re.sub('[^a-zA-Z0-9]', ' ', str(df[i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word)
 for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
  return corpus


def Extract_CountVectorizer_Features(cv,q):
    return cv.transform(q).toarray()
    
def Extract_Features_Training(X):
  cv = CountVectorizer(max_features = 5000, ngram_range=(1,2))
  X = cv.fit_transform(X).toarray()  
  return X,cv

def Train_Model(Path):
    X_text1, X_title1, y1 = datasets(Path)
    corpus1_text = simplify(X_text1, y1)
    X_train_text1, X_test_text1, y_train_text1, y_test_text1 = train_test_split(corpus1_text, y1, test_size=0.3, random_state=42) 
    X_train_text1,cv = Extract_Features_Training(X_train_text1)
    X_test_text1 = Extract_CountVectorizer_Features(cv,X_test_text1)
    clf=train(X_train_text1,y_train_text1)
    report={"Data Url:":"https://www.kaggle.com/c/fake-news/data"}
    report=model_eval(report, clf, X_test_text1, y_test_text1)
    # save the model to disk
    filename = 'Model1_LogisticRegression.sav'
    joblib.dump(clf, filename)
    # save the CountVectorizer to disk
    filename_cv = 'CountVectorizer.sav'
    joblib.dump(cv, filename_cv)

    # save report in json file
    with open('Model1_LogisticRegression_Report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print("Training completed successfully.........")
    
    
def train(x_train, y_train):
    clf = LR(max_iter=10)
    clf.fit(x_train, y_train)
    return clf
    
def model_eval(report, clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = score(y_test, y_pred,average='weighted')
    
    print(report)
    report["accuracy"]=acc
    report["precision"]=precision
    report["recall"]=recall
    report["F-Score"]=fscore
    return report


def predict(clf,x_query):
    y_pred=clf.predict(x_query)
    return y_pred


def Predict(q):
    q=[q]	

    Label={1: "Fake",
           0: "Genuine"}

    # Load the model from disk
    path=os.path.abspath(os.getcwd())

    #Train_Model(path+'/train.csv')

    filename = path+'/Model1_LogisticRegression.sav'
    print(filename)

    clf=joblib.load(filename)
    # Load the CountVectorizer from disk
    filename_cv = path+'/CountVectorizer.sav'
    cv=joblib.load(filename_cv)
    # Load the model report from disk

 
    # Opening JSON file
    f = open(path+'/Model1_LogisticRegression_Report.json') 
    report = json.load(f)
        
    
    print(q)  

  
    f=Extract_CountVectorizer_Features(cv,q)    
	
    r=predict(clf,f)

    report["Input"]=q[0]
    report["Prediction"]=Label[r[0]]
    report["Model"]="Logistic Regression"
    return report	


def plotting_accuracies(test_accuracy_text,test_accuracy_title,test_accuracy_tt):
  l=[test_accuracy_text,test_accuracy_title,test_accuracy_tt]
  l2=['Text','Title','Text+Title']
  d1={'Accuracy':l,'Variation':l2}
  d1=pd.DataFrame(d1)
  

  fig = px.bar(d1, y='Accuracy', x='Variation', text='Accuracy',color='Variation',title='Accuracy Analysis on train.csv')
  fig.update_traces(texttemplate='%{text}', textposition='outside')
  fig.show()
    


if __name__ == "__main__":

    Label={1: "Fake",
           0: "Genuine"}
    
    #==============================================For Training (Only one time)==============================
    #nltk.download('stopwords')
    Path="New_training_19245.csv"

   

    Train_Model(Path)
  

    #==============================================For Prediction==============================
    # Load the model from disk
    filename = 'Model1_LogisticRegression.sav'
    clf=joblib.load(filename)
    # Load the CountVectorizer from disk
    filename_cv = 'CountVectorizer.sav'
    cv=joblib.load(filename_cv)
    # Load the model report from disk

 
    # Opening JSON file
    f = open('Model1_LogisticRegression_Report.json') 
    report = json.load(f)
        
    
    q=['Manas is a good boy']
    f=Extract_CountVectorizer_Features(cv,q)    
    r=predict(clf,f)

    report["Input"]=q[0]
    report["Prediction"]=Label[r[0]]
    
    print(report)