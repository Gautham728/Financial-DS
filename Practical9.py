#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


dict = {"Name":["Siddhesh","Jayesh","Abhishekh","Gautham","Omkar","Manish","abdul","Bhagyavan","Pratik"],
        "Salary":[40000,35000,30000,50000,25000,39000,41000,55000,42000],
        "Country":["India","USA","UK","India","UK","SL","USA","India","India"]}


# In[5]:


df=pd.DataFrame(dict)
df


# In[6]:


dict=df.to_csv("salary.csv",index=False)


# In[7]:


df=pd.read_csv("salary.csv")
df


# In[8]:


avg=df["Salary"].mean()
avg


# In[9]:


median=df.Salary.median()
median


# In[10]:


mode=df.Salary.mode()
mode


# In[11]:


countrywise_sum=df.groupby(["Country"])["Salary"].sum()
countrywise_sum


# In[12]:


countrywise_count=df.groupby(["Country"]).count()
countrywise_count


# In[13]:


#variance of salaries
var=df["Salary"].var()
var


# In[14]:


#standard deviation
std=df["Salary"].std()
std


# In[15]:


#skewness
skew=df.skew(axis=0, skipna=True)
skew


# In[17]:


df=pd.read_csv("BirthWeight.csv")
df.head()


# In[18]:


df.set_index("Infant ID",inplace=True)
df


# In[19]:


df.cov()


# In[20]:


df.corr(method="pearson")


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis


# In[22]:


pd.set_option("display.max_columns",None)
pd.options.display.float_format="{:,.2f}".format


# In[23]:


data=pd.read_csv("diamonds.csv")
data


# In[24]:


des_df=data.drop(['id'],axis=1)#drop id column
for col in des_df:
    if des_df[col].dtype == 'object':
        des_df=des_df.drop([col],axis=1)  #Drops all alpha-numeric columns
        
des_r=des_df.describe() #describe() gives us mean, median, mode, etc...
des_r = des_r.rename(index={'50%':'median/50%'})
des_r


# In[25]:


#Only see won't come in exams
#Reference
var_r = des_df.var()

varlist = []
for col in des_df.columns:
    if des_df[col].dtype == 'object':
        continue
    varlist.append(round(des_df[col],5))
    
df = pd.DataFrame([varlist], columns = des_r.columns, index = ['var'])
mct = des_r.append(df)
mct


# In[ ]:




