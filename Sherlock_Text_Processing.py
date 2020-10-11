#!/usr/bin/env python
# coding: utf-8

# ## Text Processing for the Novel Sherlock Holmes

# #### Importing Libraries

# In[1]:


import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from nltk.tokenize import word_tokenize


# In[2]:


nltk.download('stopwords')


# In[3]:


nltk.download('punkt')


# In[4]:


nltk.download('averaged_perceptron_tagger')


# In[ ]:





# ####  Reading Novel and converting it into a text variable

# In[5]:


file = open(r"sherlock.txt",encoding='utf-8')
wordslist = file.read().splitlines() # to escape \n occurence
wordslist = [i for i in wordslist if i!='']
text = ""
text = text.join(wordslist)


# In[6]:


type(file)


# In[7]:


text[:2000] #first 2000 characters of out Novel T1


# In[41]:


len(text)


# In[ ]:





# ### Preprocessing

# In[8]:


#Creating a string which has all the punctuations to be removed
punctuations = '''!()-[]{};:'"\,<>./‘’?“”@#$%^&*_~'''
cleantext = ""
for char in text:
    if char not in punctuations:
        cleantext = cleantext + char
        
#Converting the text into lower case         
cleantext = cleantext.lower()


# In[9]:


cleantext[:2000]  #first 2000 characters of our clean text


# #### Visualization

# In[10]:


# Word cloud without removing stopwords
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,stopwords = {},colormap='winter').generate(cleantext) 

plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[11]:


tokens = word_tokenize(cleantext)
freq = nltk.FreqDist(tokens)
freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1],reverse=True)}
x = list(freq.keys())[:40]
y = list(freq.values())[:40]
plt.figure(figsize=(12,5))
plt.plot(x,y,c='r',lw=4,ls='-.')
plt.grid()
plt.xticks(rotation=90)
plt.title('Token Frequency (with stopwords)',size=17)
plt.xlabel('Words',size=14)
plt.ylabel('Count',size=14)
plt.show()


# In[12]:


tokens = word_tokenize(cleantext)
tokens[:15]   #first 15 tokens


# In[13]:


type(tokens)


# In[42]:


len(tokens)


# #### Removing stopwords and tokenization

# In[14]:


# Removing stopwords and storing it into finaltext
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(cleantext)
tokens_final = [i for i in tokens if not i in stop_words] # tokenising with removing stopwords
finaltext = "  "
finaltext = finaltext.join(tokens_final)


# In[15]:


finaltext[:2000] #first 2000 characters of our final text


# #### Visualization

# In[16]:


# Word cloud after removing stopwords
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,stopwords = {},colormap='winter').generate(finaltext) 

plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# #### Frequency distribution of tokens

# In[17]:





# #### PoS Tagging and Frequency Distribution of Tags on Text

# In[18]:


tagged = nltk.pos_tag(tokens) 
tagged[:15] #first 15 POS tags


# In[19]:


type(tagged)


# In[20]:


from collections import Counter
counts = Counter( tag for word,  tag in tagged)
print(counts)


# In[21]:


freq_tags = nltk.FreqDist(counts)
freq_tags = {k: v for k, v in sorted(freq_tags.items(), key=lambda item: item[1],reverse=True)}
x = list(freq_tags.keys())[:40]
y = list(freq_tags.values())[:40]
plt.figure(figsize=(12,5))
plt.plot(x,y,c='r',lw=4,ls='-.')
plt.grid()
plt.xticks(rotation=90)
plt.title('TAGs Frequency',size=17)
plt.xlabel('Tags',size=14)
plt.ylabel('Count',size=14)
plt.show()


# #### For word length vs Frequency distribution

# In[47]:


import numpy as np
bin_size=np.linspace(0,16)


# In[23]:


#Finding Wordlength and storing it as a list
wordLength = [len(r) for r in tokens]

#Plotting histogram of Word length vs Frequency
plt.hist(wordLength, bins=bin_size)
plt.xlabel('word length')
plt.ylabel('word length Frequency')
plt.title('Frequency Distribution for the book SHERLOCK')
plt.show()


# In[ ]:





# In[ ]:


#End of code :)) 

