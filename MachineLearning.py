import tkinter
from textblob import TextBlob
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pickle

main = tkinter.Tk()
main.title("Analysis of Women Safety in Indian Cities Using Machine Learning on Tweets") #designing main screen
main.geometry("1300x1200")

global filename
tweets_list = []
clean_list = []
global pos, neu, neg, genuine_count, fake_count

global decision_tree, tfidf

#declaring NLTK word stemmer and stop word list
SW_array = set(stopwords.words('english'))
word_stemmer = PorterStemmer()

#function to remove stop words
def removeStopWords(data):
    content = []
    for i in range(len(data)):
        if data[i] not in SW_array and len(data[i]) > 2:
            content.append(word_stemmer.stem(data[i].strip()))
    content = ' '.join(content)        
    return content

with open('model/dt.txt', 'rb') as file:
    decision_tree = pickle.load(file)
file.close()

with open('model/tfidf.txt', 'rb') as file:
    tfidf = pickle.load(file)
file.close()

def tweetCleaning(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens) #here upto for word based
    return tokens

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def read():
    text.delete('1.0', END)
    tweets_list.clear()
    train = pd.read_csv(filename,encoding='iso-8859-1')
    for i in range(len(train)):
        tweet = train.get_value(i, 'Text')
        tweets_list.append(tweet)
        text.insert(END,tweet+"\n")
    text.insert(END,"\n\nTotal tweets found in dataset is : "+str(len(tweets_list))+"\n\n\n")        

def clean():
    text.delete('1.0', END)
    clean_list.clear()
    for i in range(len(tweets_list)):
        tweet = tweets_list[i]
        tweet = tweet.strip("\n")
        tweet = tweet.strip()
        tweet = tweetCleaning(tweet.lower())
        clean_list.append(tweet)
        text.insert(END,tweet+"\n")
    text.insert(END,"\n\nTotal tweets found in dataset is : "+str(len(clean_list))+"\n\n\n")         
    
    
def checkTweet(tweet):
    global decision_tree, tfidf
    data = tweet.strip("\n").strip().lower()
    data = data.replace("\n"," ")
    data = re.sub(r'[^a-zA-Z\s]+', '', data)
    data = removeStopWords(data.split(" "))
    data = tfidf.transform([data])
    predict = decision_tree.predict(data)
    result = None
    if predict[0] == 0:
        result = "Genuine Tweet"
    if predict[0] == 1:
        result = "Fake Tweet"
    return result    

def machineLearning():
    text.delete('1.0', END)
    global pos, neu, neg, genuine_count, fake_count
    pos = 0
    neu = 0
    neg = 0
    genuine_count = 0
    fake_count = 0
    for i in range(0,100):
        tweet = clean_list[i]
        blob = TextBlob(tweet)
        tweet_type = checkTweet(tweet)
        if tweet_type == 'Genuine Tweet':
            genuine_count = genuine_count + 1
        if tweet_type == 'Fake Tweet':
            fake_count = fake_count + 1
        if blob.polarity <= 0.2:
            neg = neg + 1
            text.insert(END,tweet+"\n")
            text.insert(END,"Predicted Sentiment : NEGATIVE\n")
            text.insert(END,"Polarity Score      : "+str(blob.polarity)+"\n")
            text.insert(END,"Tweet Predicted As  : "+tweet_type+"\n")
            text.insert(END,'====================================================================================\n')
        if blob.polarity > 0.2 and blob.polarity <= 0.5:
            neu = neu + 1
            text.insert(END,tweet+"\n")
            text.insert(END,"Predicted Sentiment : NEUTRAL\n")
            text.insert(END,"Polarity Score      : "+str(blob.polarity)+"\n")
            text.insert(END,"Tweet Predicted As  : "+tweet_type+"\n")
            text.insert(END,'====================================================================================\n')
        if blob.polarity > 0.5:
            pos = pos + 1
            text.insert(END,tweet+"\n")
            text.insert(END,"Predicted Sentiment : POSITIVE\n")
            text.insert(END,"Polarity Score      : "+str(blob.polarity)+"\n")
            text.insert(END,"Tweet Predicted As  : "+tweet_type+"\n")
            text.insert(END,'====================================================================================\n')
        text.update_idletasks()    
    
def graph():
    label_X = []
    category_X = []
    text.delete('1.0', END)
    text.insert(END,"Saftey Factor\n\n")
    text.insert(END,'Positive : '+str(pos)+"\n")
    text.insert(END,'Negative : '+str(neg)+"\n")
    text.insert(END,'Neutral  : '+str(neu)+"\n\n")
    text.insert(END,'Length of tweets  : '+str(len(clean_list))+"\n")
    text.insert(END,'Positive : '+str(pos)+' / '+ str(len(clean_list))+' = '+str(pos/len(clean_list))+'%\n')
    text.insert(END,'Negative : '+str(neg)+' / '+ str(len(clean_list))+' = '+str(neg/len(clean_list))+'%\n')
    text.insert(END,'Neutral  : '+str(neu)+' / '+ str(len(clean_list))+' = '+str(neu/len(clean_list))+'%\n')
    label_X.append('Positive')
    label_X.append('Negative')
    label_X.append('Neutral')
    category_X.append(pos)
    category_X.append(neg)
    category_X.append(neu)

    plt.pie(category_X,labels=label_X,autopct='%1.1f%%')
    plt.title('Women Saftey & Sentiment Graph')
    plt.axis('equal')
    plt.show()

def fakegraph():
    global genuine_count, fake_count
    height = [genuine_count, fake_count]
    bars = ('Genuine Tweets', 'Fake Tweets')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Genuine VS Fake Tweets Garph") 
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Analysis of Women Safety in Indian Cities Using Machine Learning on Tweets')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Tweets Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=370,y=100)

readButton = Button(main, text="Read Tweets", command=read)
readButton.place(x=50,y=150)
readButton.config(font=font1) 

cleanButton = Button(main, text="Tweets Cleaning", command=clean)
cleanButton.place(x=210,y=150)
cleanButton.config(font=font1) 

mlButton = Button(main, text="Run Machine Learning Algorithm", command=machineLearning)
mlButton.place(x=400,y=150)
mlButton.config(font=font1) 

graphButton = Button(main, text="Women Saftey Graph", command=graph)
graphButton.place(x=730,y=150)
graphButton.config(font=font1)

graph1Button = Button(main, text="Genuine & Fake Tweet Graph", command=fakegraph)
graph1Button.place(x=980,y=150)
graph1Button.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
