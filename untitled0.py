import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
##from nltk.stem import PorterStemmer
import re

para ="""The world must be made safe for democracy. Its peace must be planted upon the tested foundations of political liberty. We have no selfish ends to serve. We desire no conquest, no dominion. We seek no indemnities for ourselves, no material compensation for the sacrifices we shall freely make. We are but one of the champions of the rights of mankind. We shall be satisfied when those rights have been made as secure as the faith and the freedom of nations can make them.

Just because we fight without rancor and without selfish object, seeking nothing for ourselves but what we shall wish to share with all free peoples, we shall, I feel confident, conduct our operations as belligerents without passion and ourselves observe with proud punctilio the principles of right and of fair play we profess to be fighting for.

…

It will be all the easier for us to conduct ourselves as belligerents in a high spirit of right and fairness because we act without animus, not in enmity toward a people or with the desire to bring any injury or disadvantage upon them, but only in armed opposition to an irresponsible government which has thrown aside all considerations of humanity and of right and is running amuck. We are, let me say again, the sincere friends of the German people, and shall desire nothing so much as the early reestablishment of intimate relations of mutual advantage between us—however hard it may be for them, for the time being, to believe that this is spoken from our hearts.

We have borne with their present government through all these bitter months because of that friendship—exercising a patience and forbearance which would otherwise have been impossible. We shall, happily, still have an opportunity to prove that friendship in our daily attitude and actions toward the millions of men and women of German birth and native sympathy who live among us and share our life, and we shall be proud to prove it toward all who are in fact loyal to their neighbors and to the government in the hour of test. They are, most of them, as true and loyal Americans as if they had never known any other fealty or allegiance. They will be prompt to stand with us in rebuking and restraining the few who may be of a different mind and purpose. If there should be disloyalty, it will be dealt with with a firm hand of stern repression; but, if it lifts its head at all, it will lift it only here and there and without countenance except from a lawless and malignant few.

It is a distressing and oppressive duty, gentlemen of the Congress, which I have performed in thus addressing you. There are, it may be, many months of fiery trial and sacrifice ahead of us. It is a fearful thing to lead this great peaceful people into war, into the most terrible and disastrous of all wars, civilization itself seeming to be in the balance. But the right is more precious than peace, and we shall fight for the things which we have always carried nearest our hearts—for democracy, for the right of those who submit to authority to have a voice in their own governments, for the rights and liberties of small nations, for a universal dominion of right by such a concert of free peoples as shall bring peace and safety to all nations and make the world itself at last free.

To such a task we can dedicate our lives and our fortunes, everything that we are and everything that we have, with the pride of those who know that the day has come when America is privileged to spend her blood and her might for the principles that gave her birth and happiness and the peace which she has treasured. God helping her, she can do no other."""
##ps = PorterStemmer()
wordnet = WordNetLemmatizer()

sentences= nltk.sent_tokenize(para)

corpus=[]

for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ' , sentences[i])
    review=review.lower()
    review= review.split()
    review=[wordnet.lemmatize(word)for word in review if not word in set(stopwords.words('English'))]
    review=' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv= TfidfVectorizer()
x= cv.fit_transform(corpus).toarray() 
    
    
    