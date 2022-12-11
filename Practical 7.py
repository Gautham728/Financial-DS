#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Uniform Ditribution
from numpy import random as r
import matplotlib.pyplot as plt
import seaborn as sns

UniformMatrix = r.uniform(0.2, 0.4, size = (10))

print(UniformMatrix)


# In[2]:


sns.distplot(r.uniform(size = (1000)), hist = False)


# # Bernoulli Distribution

# In[3]:


from scipy.stats import bernoulli
data_bern = bernoulli.rvs(size = 10000, p = 0.6)


# In[4]:


ax = sns.distplot(data_bern, kde = False, color = "skyblue", hist_kws = {"linewidth" : 15, "alpha" : 1})
ax.set(xlabel = "Bernoulli Distribution", ylabel = "Frequency")
     


# # Binomial Distribution

# In[5]:


from scipy.stats import binom
data_binom = binom.rvs(n = 10, p = 0.8, size = 10000)
     


# In[6]:


ax = sns.distplot(data_binom, kde = False, color = "skyblue", hist_kws = {"linewidth" : 15, "alpha" : 1})
ax.set(xlabel = "Binomial Distribution", ylabel = "Frequency")


# # Poission Distribution

# In[7]:


from scipy.stats import poisson
data_poisson = poisson.rvs(mu = 3, size = 10000)
     


# In[8]:


ax = sns.distplot(data_poisson, bins = 30, kde = False, color = "skyblue", hist_kws = {"linewidth" : 15, "alpha" : 1})
ax.set(xlabel = "Poisson Distribution", ylabel = "Frequency")


# In[9]:


#A ware house typically recieves 8 delieveres between 4 and 5 on Friday
#What is the probability that only 4 delieveries will arrive between 4 and 5 pm on friday?
from scipy.stats import poisson
poisson.pmf(4,8)


# In[10]:


#What is the probability of setting less than = 3 delieveries on friday between 4 and 5 pm?
from scipy.stats import poisson
poisson.cdf(3,8)


# In[11]:


#What is the probability of having no delieveries on friday between 4 and 5 pm?
from scipy.stats import poisson
poisson.cdf(0,8)


# In[ ]:




