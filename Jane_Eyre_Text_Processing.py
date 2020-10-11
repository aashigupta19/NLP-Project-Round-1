#!/usr/bin/env python
# coding: utf-8

# ## Text Processing for the Novel Jane Eyre

# #### Importing Libraries

# In[1]:


import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from nltk.tokenize import word_tokenize


# In[ ]:





# In[2]:


nltk.download('stopwords')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')


# #### Reading Novel and Converting it in Text Variable

# In[3]:


file_eyre = open(r"jane_eyre.txt",encoding='utf-8')
wordslist_eyre = file_eyre.read().splitlines() # to escape \n occurence
wordslist_eyre = [i for i in wordslist_eyre if i!='']
text_eyre = ""
text_eyre = text_eyre.join(wordslist_eyre)


# In[4]:


text_eyre[:2000] #first 2000 characters of the novel


# In[8]:


len(text_eyre)


# ### Preprocessing

# In[6]:


#Creating a string which has all the punctuations to be removed
punctuations_eyre = '''!()-[]{};:'"\,<>./‘’?“”@#$%^&*_~'''
cleantext_eyre = ""
for char in text_eyre:
    if char not in punctuations_eyre:
        cleantext_eyre = cleantext_eyre + char
        
#Converting the text into lower case         
cleantext_eyre = cleantext_eyre.lower()


# In[7]:


cleantext_eyre[:2000] #first 2000 characters of clean text


# In[ ]:





# In[9]:


tokens_eyre_t = word_tokenize(cleantext_eyre)
tokens_eyre_t[:15] #first 15 tokens


# In[10]:


type(tokens_eyre_t)


# In[11]:


len(tokens_eyre_t)


# #### Visualization

# In[12]:


# Word cloud without removing stopwords
wordcloud_eyre_withStopwords = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,stopwords = {},colormap='winter').generate(cleantext_eyre) 

plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud_eyre_withStopwords) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[13]:


#Frequency Distribution of Token with StopWords

tokens_eyre = word_tokenize(cleantext_eyre)
freq_eyre = nltk.FreqDist(tokens_eyre)
freq_eyre = {k: v for k, v in sorted(freq_eyre.items(), key=lambda item: item[1],reverse=True)}
x = list(freq_eyre.keys())[:40]
y = list(freq_eyre.values())[:40]
plt.figure(figsize=(12,5))
plt.plot(x,y,c='r',lw=4,ls='-.')
plt.grid()
plt.xticks(rotation=90)
plt.title('Token Frequency (with stopwords)',size=17)
plt.xlabel('Words',size=14)
plt.ylabel('Count',size=14)
plt.show()


# #### Removing stopwords and Tokenization

# In[14]:


# Removing stopwords and storing it into finaltext
stop_words_eyre = set(stopwords.words('english'))
tokens_eyre = word_tokenize(cleantext_eyre)
tokens_final_eyre = [i for i in tokens_eyre if not i in stop_words_eyre] # tokenising with removing stopwords
finaltext_eyre = "  "
finaltext_eyre = finaltext_eyre.join(tokens_final_eyre)

finaltext_eyre[:2000] #first 2000 characters of final text


# #### Visualization

# In[15]:


# Word cloud with removing stopwords
wordcloud_eyre_withoutStopWords = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,stopwords = {},colormap='winter').generate(finaltext_eyre) 

plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud_eyre_withoutStopWords) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[21]:


#Frequency Distribution of Tokens without StopWords

tokens_eyre = word_tokenize(finaltext_eyre)
tokens_eyre = [i for i in tokens_eyre if not i in stop_words_eyre]
freq_eyre = nltk.FreqDist(tokens_eyre)
freq_eyre = {k: v for k, v in sorted(freq_eyre.items(), key=lambda item: item[1],reverse=True)}
x = list(freq_eyre.keys())[:40]
y = list(freq_eyre.values())[:40]
plt.figure(figsize=(12,5))
plt.plot(x,y,c='r',lw=4,ls='-.')
plt.grid()
plt.xticks(rotation=90)
plt.title('Token Frequency (without stopwords)',size=17)
plt.xlabel('Words',size=14)
plt.ylabel('Count',size=14)
plt.show()


# In[ ]:





# ### PoS Tagging and Frequency Distribution of Tags over Text

# In[17]:


tagged_eyre = nltk.pos_tag(tokens_eyre) 
tagged_eyre[:15] #first 15 tags


# In[20]:


type(tagged_eyre)


# In[ ]:





# In[19]:


from collections import Counter
counts_eyre = Counter( tag for word,  tag in tagged_eyre)
print(counts_eyre)


# In[ ]:





# In[22]:


#Frequency Distribution of PoS Tags

freq_tags_eyre = nltk.FreqDist(counts_eyre)
freq_tags_eyre = {k: v for k, v in sorted(freq_tags_eyre.items(), key=lambda item: item[1],reverse=True)}
x = list(freq_tags_eyre.keys())[:40]
y = list(freq_tags_eyre.values())[:40]
plt.figure(figsize=(12,5))
plt.plot(x,y,c='r',lw=4,ls='-.')
plt.grid()
plt.xticks(rotation=90)
plt.title('Frequency Distribution of PoS Tags',size=17)
plt.xlabel('Words',size=14)
plt.ylabel('Count',size=14)
plt.show()


# In[ ]:





# ### For Word length vs Frequency Relation

# In[23]:


import numpy as np
bin_size=np.linspace(0,16)


# In[24]:


#Finding Wordlength and storing it as a list
wordLength_eyre = [len(r) for r in tokens_eyre]

#Plotting histogram of Word length vs Frequency
plt.hist(wordLength_eyre, bins=bin_size)
plt.xlabel('word length')
plt.ylabel('word length Frequency')
plt.title('Frequency Distribution for the book Jane Eyre')
plt.show()


# In[ ]:





# In[25]:


# End of Code :)) 


# In[26]:





# In[ ]:




