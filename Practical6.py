#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

n = np.random.randint(2,10,50)
print(n)


# In[2]:


df = pd.DataFrame(n)
df = pd.DataFrame(df[0].value_counts())
df


# In[3]:


length = len(n)
print(length)


# In[4]:


df.columns = ["counts"]
df


# In[5]:


#to calculate probability mass function
df["prob"] = df["counts"]/length
df


# In[6]:


#fig = plt.figure(figsize = (10,5))
plt.bar(df["counts"],df["prob"])
plt.xlabel("counts")
plt.ylabel("pmf")
plt.title("Probability Mass Function")
plt.show()


# In[7]:


import seaborn as sns
sns.barplot(x = 'counts',y = 'prob',data = df)


# In[8]:


sns.barplot(df["counts"],df["prob"])


# In[9]:


#another example for pmf

data = {'Candy':['Blue','Orange','Green','Red'],
       'Total':[10,50,20,35]}
df = pd.DataFrame(data)
df


# In[10]:


df["pmf"] = df["Total"] / df["Total"].sum()
df


# In[11]:


plt.bar(data["Candy"],data["Total"])
plt.xlabel("Candy")
plt.ylabel("Total")
plt.title("Probability Mass Function")
plt.show()


# In[12]:


sns.barplot(x=df["Candy"],y=df["Total"])


# In[13]:


data = np.random.normal(size = 100)
data = np.append(data, [1.2,1.2,1.2,1.2,1.2])
sns.distplot(data)
sns.displot(data)


# In[14]:


import scipy.stats as stats

mu = 20
sigma = 2
h = sorted(np.random.normal(mu, sigma, 100))


# In[15]:


import scipy.stats as stats
plt.figure(figsize = (10,5))

fit = stats.norm.pdf(h, np.mean(h), np.std(h))

plt.plot(h, fit, '-o')

plt.hist(h, density = True)


# In[16]:


import scipy.stats as ss

x = np.linspace(-5,5,5000)
mu = 0
sigma = 1

y_pdf = ss.norm.pdf(x, mu, sigma)   #the normal pdf
y_cdf = ss.norm.cdf(x, mu, sigma)   #the normal cdf

plt.plot(x, y_pdf, label = "pdf")
plt.plot(x, y_cdf, label = "cdf")


# In[17]:


import scipy.stats as stats
plt.figure(figsize = (10,5))

fit = stats.norm.cdf(h, np.mean(h), np.std(h))

plt.plot(h, fit, '-o')

plt.hist(h, density = True)


# In[ ]:




