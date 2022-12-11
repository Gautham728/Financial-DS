#!/usr/bin/env python
# coding: utf-8

# # Continious Distribution

# In[1]:


from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#generate random nos. from N(0,1)
data_normal = norm.rvs(size = 1000, loc = 0, scale = 1)


# In[2]:


ax = sns.distplot(data_normal, bins = 100, kde = True, color = "skyblue", hist_kws = {"linewidth":15, "alpha":1})
ax.set(xlabel = "Normal Distribution", ylabel = "Frequency")


# In[4]:


#import io
#3df = pd.read_csv(io.BytesIO(uploaded['/content/drive/MyDrive/Google Colab/weight_height.csv']))
# Dataset is now stored in a Pandas Dataframe
df=pd.read_csv("weight-height.csv.csv")
df


# In[5]:


df.Height.describe()


# In[6]:


mean = df.Height.mean()
mean


# In[7]:


std_deviation = df.Height.std()
std_deviation


# In[8]:


mean+3*std_deviation


# In[9]:


df[(df.Height<54.82) | (df.Height>77.91)]


# In[10]:


df_no_outlier = df[(df.Height<77.91) & (df.Height>54.82)]
df_no_outlier.shape


# In[11]:


df["zscore"] = (df.Height - df.Height.mean()) / df.Height.std()
df.head()


# In[12]:


df.Height.mean()


# In[13]:


df.Height.std()


# In[14]:


(77.84 - 66.37) / 3.84


# In[15]:


df[df["zscore"]>3]


# In[16]:


df[df["zscore"]<-3]


# In[17]:


get_ipython().system('pip install ipython')


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#for latex equations
from IPython.display import Math, Latex
#for displaying images
from IPython.display import Image
import numpy as np
#neglects the warnings
import warnings
warnings.filterwarnings('ignore')


# In[19]:


import seaborn as sns
#settings for seaborn plotting style
sns.set(color_codes=True)
#setting for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})


# In[20]:


#import uniform distribution
from scipy.stats import uniform


# In[21]:


#random numbers from uniform distribution
n=10000
start=10
width=20
data_uniform=uniform.rvs(size=n,loc=start, scale=width)


# In[22]:


ax=sns.distplot(data_uniform,
                bins=100,
                kde=True,
                color='skyblue',
                hist_kws={"linewidth":15,"alpha":1})
ax.set(xlabel="Uniform Distribution",ylabel="Frequency")


# In[23]:


from scipy.stats import norm
#generate random numbers from n(0,1)
data_normal=norm.rvs(size=10000,loc=0,scale=1)


# In[24]:


ax=sns.distplot(data_normal,
                bins=100,
                kde=True,
                color='skyblue',
                hist_kws={"linewidth":15,"alpha":1})
ax.set(xlabel="Normal Distribution",ylabel="Frequency")


# In[25]:


from scipy.stats import expon
data_expon=expon.rvs(size=10000,loc=0,scale=1)


# In[26]:


ax=sns.distplot(data_expon,
                bins=100,
                kde=True,
                color='skyblue',
                hist_kws={"linewidth":15,"alpha":1})
ax.set(xlabel="Exponential Distribution",ylabel="Frequency")


# In[27]:


from numpy import random
x=random.chisquare(df=2,size=(2,3))
print(x)


# In[28]:


#import numpy, seaborn, matpltlib.pyplot
sns.distplot(random.chisquare(df=1,size=1000),hist=False)
plt.show()


# In[29]:


a=5.
s=np.random.weibull(a,1000)


# In[30]:


#import matplot
x=np.arange(1,100.)/50.
def weib(x,n,a):
    return (a/n)*(x/n)**(a-1)*np.exp(-(x/n)**a)


# In[31]:


count, bins, ignored=plt.hist(np.random.weibull(5.,1000))
x=np.arange(1,100.)/50.
scale=count.max()/weib(x,1.,5.).max()
plt.plot(x,weib(x,1.,5.)*scale)
plt.show()


# In[ ]:




