#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import sem,t
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("loans_income.csv")
df


# In[4]:


loans_income=np.array(pd.read_csv("loans_income.csv"))
loans_income[:5]


# In[5]:


#Making a flat list from list of lists
loans_income=np.array([item for sublist in loans_income for item in sublist])


# In[6]:


def bootstrap(l,R):
  n=len(loans_income)
  #Number of bootstrap samples
  means_of_boot_samples=[]
  for steps in range(R):
    #Steps 1,2
    boot_sample=np.random.choice(loans_income,size=n)
    #step 3
    means_of_boot_samples.append(round(np.mean(boot_sample),3))
  return means_of_boot_samples

bootstrap(loans_income,5)


# In[7]:


np.std(bootstrap(loans_income,100))


# In[12]:


#Plot histogram
plt.figure(dpi=200)

plt.subplot(221)
plt.title("R = 10.000")
plt.hist(bootstrap(loans_income, 10000),edgecolor="k")

plt.subplot(222)
plt.title("R = 1000")
plt.hist(bootstrap(loans_income, 1000),edgecolor="k")

plt.subplot(223)
plt.title("R = 100")
plt.hist(bootstrap(loans_income, 100),edgecolor="k")

plt.subplot(224)
plt.title("R = 10")
plt.hist(bootstrap(loans_income, 10),edgecolor="k")

plt.tight_layout()


# In[9]:


#Find a confidence interval
data=bootstrap(loans_income,1000)
lower_lim, upper_lim = np.percentile(data,2.5), np.percentile(data, 95)
print("Lower Limit: ", lower_lim)
print("Upper Limit: ", upper_lim)


# In[11]:


plt.figure(dpi=200)
plt.title("95% Confidence interval of loan applicants based on a sample of 1000 means")

sns.distplot(bootstrap(loans_income,1000),hist=True,kde=True,
             color="darkblue",bins=50,
             hist_kws={"edgecolor":"black"},
             kde_kws={"linewidth":2})

plt.axvline(x=lower_lim,color="red")
plt.axvline(x=upper_lim,color="red")


# In[ ]:




