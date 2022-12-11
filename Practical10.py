#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_1samp
from statsmodels.stats.power import tt_ind_solve_power
import warnings
warnings.filterwarnings("ignore")


# T test<br>
# A T test is inferentiqal statistics which is used to determine if there is a significant difference between means of two groups which may be related in certain features.<br>
# 
# T-test has 2 types<br>
# 
# One sampled t test<br>
# Two sampled t test<br>
# t=(sample mean - population mean) / standard error

# In[2]:


ages=[10,20,35,50,28,40,55,18,16,55,30,25,43,18,30,28,14,24,16,17,32,35,26,27,65,18,43,23,21,20,19,70]
ages_mean=np.mean(ages)
print(ages_mean)


# In[3]:


#lets take sample
sample_size=10
age_sample=np.random.choice(ages,sample_size)
age_sample


# In[4]:


from scipy.stats import ttest_1samp


# In[5]:


ttest,p_value=ttest_1samp(age_sample,30)


# In[6]:



print(p_value)


# In[7]:


if p_value<0.05:
   print("We are rejecting null hypothesis")
else:
  print("We are accepting null hypothesis")


# In[8]:



df=pd.read_excel("Result.xlsx")
df


# In[9]:


df.describe()


# In[11]:


#One way Hypothesis

Ho="mu <= 113"
Ha="mu > 113"    #alternate hypothesis
al=0.05          #ALPHA VALUE
#mu -> mean
mu=113
#tail type
tt=1
#data
marks=df["Total"].values
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
print("mu:", mu)
print(marks)
print("")


# In[12]:


ts, pv = ttest_1samp(marks,mu)
print("t-stat",ts)
print("p-vals",pv)
t2pv=pv
t1pv=pv*2
print("1t pv",t1pv)
print("2t pv",t2pv)


# In[13]:


if tt==1:
  if t1pv < al:
    print("Null Hypothesis: Rejected")
    print("Conclusion:",Ha)
  else:
    print("Null Hypothesis: Not Rejected")
    print("Conclusion:",Ho)
else:
  if t2pv < al/2:
    print("Null Hypothesis: Rejected")
    print("Conclusion:",Ha)
  else:
    print("Null Hypothesis: Not Rejected")
    print("Conclusion:",Ho)


# In[14]:


#problem : check if the total mean marks = 113
#Test: One sample mean without std dev

#null hyp
Ho="mu = 113"
Ha="mu =! 113"    #alternate hypothesis
al=0.05          #ALPHA VALUE
#mu -> mean
mu=113
#tail type
tt=2
#data
marks=df["Total"].values
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
print("mu:", mu)
print(marks)
print("")


# In[15]:


ts, pv = ttest_1samp(marks,mu)
print("t-stat",ts)
print("p-vals",pv)
t2pv=pv
t1pv=pv*2
print("1t pv",t1pv)
print("2t pv",t2pv)


# In[16]:


if tt==1:
  if t1pv < al:
    print("Null Hypothesis: Rejected")
    print("Conclusion:",Ha)
  else:
    print("Null Hypothesis: Not Rejected")
    print("Conclusion:",Ho)
else:
  if t2pv < al/2:
    print("Null Hypothesis: Rejected")
    print("Conclusion:",Ha)
  else:
    print("Null Hypothesis: Not Rejected")
    print("Conclusion:",Ho)


# In[17]:


subj1=np.array([45,36,29,40,46,37,43,39,28,33])
subj2=np.array([40,20,30,35,29,43,40,39,28,31])


# In[18]:



sns.distplot(subj1)


# In[19]:



sns.distplot(subj2)


# In[20]:


t_stat, p_val=stats.ttest_ind(subj1,subj2)
t_stat, p_val


# In[21]:


#perfgorm two sample t-test with equal variances
stats.ttest_ind(subj1,subj2,equal_var=True)


# In[22]:


effect_size=abs((subj1.mean()-subj2.mean())/(subj1.std()-subj2.std()))
sample_size=10
alpha=0.05
ratio=1.0

statistical_power=tt_ind_solve_power(effect_size=effect_size, nobs1=sample_size, alpha=alpha, ratio=1.0, alternative="two-sided")
print(statistical_power)


# In[23]:


type_2_error=1-statistical_power
type_2_error


# In[ ]:




