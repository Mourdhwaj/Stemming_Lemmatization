import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """Most people are not very familiar with the concept of artificial intelligence (AI). As an illustration, when 1,500 senior business leaders in the United States in 2017 were asked about AI, only 17 percent said they were familiar with it.1 A number of them were not sure what it was or how it would affect their particular companies. They understood there was considerable potential for altering business processes, but were not clear how AI could be deployed within their own organizations.

Despite its widespread lack of familiarity, AI is a technology that is transforming every walk of life. It is a wide-ranging tool that enables people to rethink how we integrate information, analyze data, and use the resulting insights to improve decisionmaking. Our hope through this comprehensive overview is to explain AI to an audience of policymakers, opinion leaders, and interested observers, and demonstrate how AI already is altering the world and raising important questions for society, the economy, and governance.

In this paper, we discuss novel applications in finance, national security, health care, criminal justice, transportation, and smart cities, and address issues such as data access problems, algorithmic bias, AI ethics and transparency, and legal liability for AI decisions. We contrast the regulatory approaches of the U.S. and European Union, and close by making a number of recommendations for getting the most out of AI while still protecting important human values.2

In order """

#preprocessing the data

text= re.sub(r'\[[0-9]*\]', ' ', paragraph)
text= re.sub(r'\s+',' ',text)
text=text.lower()
text= re.sub(r'\d',' ',text)
text = re.sub (r'\s+',' ', text)

#preprocessing the dataset

sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    
    sentences[i]= [word for word in sentences[i] if word not in stopwords.words('english')]
    
#Training the Word2Vec model

model= Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

#words = model.wv.vocab

#finding the Word Vectors

vector = model.wv.get_vector('finance')

print(vector)

#Most Similar word

similar = model.wv.most_similar('finance')

print(similar)
























   
