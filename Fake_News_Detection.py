#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import joblib
import string

#read data
fake=pd.read_csv("Fake.csv")
true=pd.read_csv("True.csv")

fake.head()

true.head()

fake['class']=0
true['class']=1

data =pd.concat([fake,true],axis=0)


data.sample(5)


data = data.drop(["title","subject","date"],axis=1)

data.reset_index(inplace=True)

data.drop(['index'],axis=1,inplace=True)

x=data.sample(5)

def clean_text(text):
    text=text.lower()
    text=re.sub('\[.*?\]',"",text)
    text=re.sub("\\W"," ",text)
    text=re.sub("https?:://\S+|WWW\.\S+","",text)
    text=re.sub("<.*?>+","",text)
    text=re.sub("[%s]" % re.escape(string.punctuation),"",text)
    text=re.sub("\n","",text)
    text=re.sub("\w\d\w*","",text)
    return text

data["text"]=data["text"].apply(clean_text)

x=data["text"]
y=data["class"]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=42)

vectorizer=TfidfVectorizer()
xv_train=vectorizer.fit_transform(x_train)
xv_test=vectorizer.transform(x_test)

#CREATE THE MODEL

model=LogisticRegression()
model.fit(xv_train,y_train)


y_pred=model.predict(xv_test)

model.score(xv_test,y_test)


print(classification_report(y_test,y_pred))

joblib.dump(vectorizer,"Vector.jb")
joblib.dump(model,"Model.jb")


