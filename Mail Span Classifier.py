import pandas as pd

messages= pd.read_csv('C:\\Users\\dhruv\\Downloads\\sms+spam+collection\\SMSSpamCollection',sep='\t',names=["label","message"])

import re
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#ps= PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordNet= WordNetLemmatizer()

corpus=[]

for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review= review.split()
    review= [wordNet.lemmatize(word) for word in review if not word in stopwords.words('english')]
 #  review= [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
    
    
"""
    Implementing Bag of Word
   
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
X= cv.fit_transform(corpus).toarray()    
"""
 
#Implenting TF IDF 

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X= cv.fit_transform(corpus).toarray()



y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=0)


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred= spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)













