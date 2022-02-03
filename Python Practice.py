#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello World!") # this is in a code cell


# In[2]:


type("Hello World!")


# In[3]:


for x in range(10): # a colon denotes the start of an indented block
    if x < 5: # Jupyter automatically indent by four spaces
        print(x)
    else:
        pass # do nothing


# In[5]:


for x in range(100): 
    if x > 90: 
        print(x)
    else:
        pass 


# In[7]:


for x in range(10):
    if x < 5:
        print(x)
    else:
        pass


# In[8]:


for x in range(10):
if x < 5:
print(x)
else:
pass #This statement won't work if we mess up with indentations


# In[9]:


# to be able to do mathematical operations, we need to convert strings to numbers...
wrong_data = '100' # error if you do numerical opertion on a string type

right_data = int(wrong_data) # convert to float

print(right_data, type(right_data))


# In[10]:



wrong_data = '100' 

right_data = int(wrong_data) 

print(right_data, type(right_data))


# In[11]:


value = None # True & False are special kinds of data
print(value, type(value))


# In[12]:


x = -0.05

# the 1st and 2th argument is for how many decimals you need for the results
print(round(x), round(x, 1), round(x, 2))


# In[15]:


x = -0.05
print(round(x), round(x, 1), round(x, 2), round(x, 3))


# In[16]:


x = 1.595865
print(round(x), round(x, 1), round(x, 2), round(x, 3))


# In[17]:


x = 1.595865
print(round(x), round(x, 1), round(x, 2), round(x, 4))


# In[18]:


# create a variable store name
# input will return whatever input by user.
print("What's your name?")
my_name = input()
print("What's your age?")
# input function outputs string now
my_age = input()
print("Hello, my name is {} and I am {} years old!".format(my_name, my_age))


# In[19]:



print("What's your name?")
my_name = input()
print("What's your age?")
my_age = input()
print("Hello, my name is {} and I am {} years old!".format(my_name, my_age))


# In[25]:


import pandas as pd
df = pd.read_csv('AAPL.csv')
df.head()


# In[26]:


df.tail()


# In[27]:


df.tail(2)


# In[28]:


df.head(2)


# In[29]:


import numpy as np
normal_return = []
for i in range(0,len(df)-1):
    adjclose_yesterday = df.iloc[i]['Adj Close']
    adjclose_today = df.iloc[i+1]['Adj Close']
    x = (adjclose_today - adjclose_yesterday) / adjclose_yesterday
    normal_return.append(x)
normal_return[:5]


# In[30]:


log_return = []
for i in range(0,len(df)-1):
    adjclose_yesterday = df.iloc[i]['Adj Close']
    adjclose_today = df.iloc[i+1]['Adj Close']
    y = np.log(adjclose_today / adjclose_yesterday)
    log_return.append(y)
log_return[:5]


# In[31]:


dfnr = pd.DataFrame(normal_return, columns = ['normal']) 
nr = dfnr.mean() * len(dfnr)
nv = dfnr.std() * (len(dfnr) ** 0.5)
print('The annulized normal return is %.8f and its annulized volatility is %.8f' % (nr,nv))


# In[32]:


dflr = pd.DataFrame(log_return, columns = ['log']) 
lr = dflr.mean() * len(dflr)
lv = dflr.std() * (len(dflr) ** 0.5)
print('The annulized log return is %.8f and its annulized volatility is %.8f' % (lr,lv))


# In[33]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(df['Close'])
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Closed Price');


# In[34]:


fig = plt.figure()
plt.plot(dflr * 100)
plt.xlabel('Days')
plt.ylabel('Percentage % ')
plt.title('Log Return');


# In[35]:


fig = plt.figure()
plt.plot(df['Close'])
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Closed Price');


# In[36]:


strike = np.array([100, 110, 120, 130, 140])
ttm = np.array([1/52, 2/52, 1/12, 3/12, 7/12])
strike, ttm = np.meshgrid(strike, ttm)
v1 = np.array([0.4901, 0.2682, 0.1995, 0.2637, 0.3742])
v2 = np.array([0.0009, 0.1912, 0.1954, 0.2296, 0.2981])
v3 = np.array([0.0000, 0.0000, 0.1976, 0.2139, 0.2568])
v4 = np.array([0.2752, 0.2370, 0.2223, 0.2214, 0.2305])
v5 = np.array([0.3093, 0.2964, 0.2891, 0.2872, 0.2858])
iv = np.array([v1, v2, v3, v4, v5])
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize= (10, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(strike, ttm, iv, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[37]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 60)
ax.scatter(strike, ttm, iv, zdir='z', s=25, c='b', marker='^')
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility');


# In[44]:


import pandas as pd
df = pd.read_csv('AAPL.csv')
df.head(10)


# In[48]:


df = pd.read_csv('AAPL.csv')
df.head()


# In[60]:


pip install yfinance


# In[1]:


import yfinance as yf
import numpy as np
import pandas as pd


# In[5]:


data = yf.download("AMZN TSLA", start="2021-02-01", end="2022-02-01")


# In[3]:


data.tail()


# In[6]:


ADJ =  data['Adj Close']
ADJ.info()


# In[7]:


ADJ.describe().round(2)


# In[8]:


data = yf.download("AMZN TSLA AAPL", start="2021-02-01", end="2022-02-01")


# In[9]:


data.head()


# In[17]:


ADJ.describe().round(2)


# In[18]:


data = yf.download("FB", start="2021-02-01", end="2022-02-01")


# In[25]:


data.head(100)


# In[8]:


data = yf.download("AMZN TSLA AAPL", start="2021-02-01", end="2022-02-01")


# In[9]:


data.head(10)


# In[10]:


sym = 'AMZN'
AMZN_data = pd.DataFrame(ADJ[sym]).dropna()
AMZN_data.tail()


# In[11]:


ADJ =  data['Adj Close']
ADJ.info()


# In[12]:


ADJ.plot(figsize=(10, 12), subplots=True)


# In[13]:


ADJ.describe().round(2)


# In[14]:


rets = np.log(ADJ / ADJ.shift(1))
rets.head().round(4)


# In[15]:


rets.cumsum().apply(np.exp).plot(figsize=(10, 6));


# In[17]:


sym = 'AMZN'
AMZN_data = pd.DataFrame(ADJ[sym]).dropna()
AMZN_data.tail()


# In[18]:


AMZN_data['SMA1'] = ADJ[sym].rolling(window=20).mean()
AMZN_data['SMA2'] = ADJ[sym].rolling(window=60).mean()
AMZN_data[[sym, 'SMA1', 'SMA2']].tail()


# In[19]:


AMZN_data.dropna(inplace=True)
AMZN_data['positions'] = np.where(AMZN_data['SMA1'] > AMZN_data['SMA2'],1,-1)
ax = AMZN_data[[sym, 'SMA1', 'SMA2', 'positions']].plot(figsize=(10, 6),secondary_y='positions')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))


# In[20]:


rets.dropna(inplace=True)
rets.plot(subplots=True, figsize=(10, 6))


# In[21]:


pd.plotting.scatter_matrix(rets, alpha=0.2, diagonal='hist', hist_kwds={'bins': 35}, figsize=(10, 6))


# In[22]:


reg = np.polyfit(rets['AMZN'], rets['TSLA'], deg=1)
ax = rets.plot(kind='scatter', x='AMZN', y='TSLA', figsize=(10, 6))
ax.plot(rets['AMZN'], np.polyval(reg, rets['AMZN']), 'r', lw=2);


# In[23]:


ax = rets['AMZN'].rolling(window=20).corr(rets['TSLA']).plot(figsize=(10, 6)) 
ax.axhline(rets.corr().iloc[0, 1], c='r');


# In[ ]:




