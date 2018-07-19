# -*- coding: utf-8 -*-
# In[1]
import os
import utils
from gensim import corpora, models

utils.echo()
# In[2]
if(not(os.path.exists('./data'))):
    os.mkdir('./data')

if(not(os.path.exists('./intermediary'))):
    os.mkdir('./intermediary')

# In[3]
raw_data = utils.preprocess.read_data_all()

# In[4]
process_data = utils.preprocess.cut_and_delete(raw_data)

# In[5]
dictionary = corpora.Dictionary(process_data)
corpus = [ dictionary.doc2bow(train) for train in process_data ]
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)

# In[6]
print(lda.print_topics(3))