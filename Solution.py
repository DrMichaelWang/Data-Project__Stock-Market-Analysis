
# coding: utf-8

# ### Stock Market Analysis

# #### Goal: use pandas to get stock information, visualize different aspects of it, analyze the risk of a particular stock based on its performance history, and predict future stock prices using Monte Carlo method 

# #### Basic Analysis of Stock

# In[1]:

# questions:
# what was the change of stock prices over time?
# what was the daily return of the stock on average?
# what was the moving average of the various stocks?


# In[2]:

# analyze the attributes of stock


# In[3]:

import pandas as pd


# In[4]:

from pandas import Series, DataFrame


# In[5]:

import numpy as np


# In[6]:

import matplotlib.pyplot as plt


# In[7]:

import seaborn as sns


# In[8]:

sns.set_style('whitegrid')


# In[9]:

get_ipython().magic(u'matplotlib inline')


# In[10]:

import pandas_datareader.data as web


# In[11]:

from datetime import datetime


# In[12]:

# use datetime library to set the start and end of time


# In[13]:

from __future__ import division


# In[14]:

# create a list for tech stocks (tickers)
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']


# In[15]:

# set start and end time
end = datetime.now()


# In[16]:

# set start time as a year ago
start = datetime(end.year-1, end.month, end.day)


# In[17]:

# set stock information as dataframe
for stock in tech_list:
    globals()[stock] = web.DataReader(stock, 'google', start, end)


# In[18]:

# Now let's check the extracted dataframes
AAPL


# In[19]:

GOOG


# In[20]:

MSFT


# In[21]:

AMZN


# In[22]:

# Now we have 4 stock dataframes containing price information by date: opening, high, low, closing, and adjusted closing prices 
# and volume
AAPL.describe()


# In[23]:

GOOG.describe()


# In[24]:

MSFT.describe()


# In[25]:

AMZN.describe()


# In[26]:

AAPL.info()


# In[27]:

GOOG.info()


# In[28]:

MSFT.info()


# In[29]:

AMZN.info()


# In[30]:

AAPL['Close'].plot(legend = True, figsize = (10, 4)) # 10 by 4 inches


# In[31]:

AAPL['Open'].plot(legend = True, figsize = (10, 4))


# In[32]:

AAPL['Volume'].plot(legend = True, figsize = (10,4))


# In[33]:

GOOG['Close'].plot(legend = True, figsize = (10,4))


# In[34]:

GOOG['Open'].plot(legend = True, figsize = (10,4))


# In[35]:

GOOG['Volume'].plot(legend = True, figsize = (10,4))


# In[36]:

MSFT['Close'].plot(legend = True, figsize = (10,4))


# In[37]:

MSFT['Open'].plot(legend = True, figsize = (10,4))


# In[38]:

MSFT['Volume'].plot(legend = True, figsize = (10,4))


# In[39]:

AMZN['Close'].plot(legend = True, figsize = (10,4))


# In[40]:

AMZN['Open'].plot(legend = True, figsize = (10,4))


# In[41]:

AMZN['Volume'].plot(legend = True, figsize = (10,4))


# In[45]:

# Calculate the rolling mean (moving average) to give insights on trends in data
# in this case, calculate three 
ma_day = [20, 50, 100]

for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    AAPL[column_name]=pd.rolling_mean(AAPL['Close'], ma)


# In[46]:

# plot it up
# close prices
AAPL[['Close', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[48]:

for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    AAPL[column_name]=AAPL['Close'].rolling(window= ma,center=False).mean()


# In[49]:

AAPL[['Close', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[50]:

# open prices
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    AAPL[column_name]=AAPL['Open'].rolling(window= ma,center=False).mean()

AAPL[['Open', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[51]:

# trade volumes
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    AAPL[column_name]=AAPL['Volume'].rolling(window= ma,center=False).mean()

AAPL[['Volume', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[53]:

# close prices for GOOG
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    GOOG[column_name]=GOOG['Close'].rolling(window= ma,center=False).mean()

GOOG[['Close', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[54]:

# Open prices for GOOG
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    GOOG[column_name]=GOOG['Open'].rolling(window= ma,center=False).mean()

GOOG[['Open', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[55]:

# trading volumes for GOOG
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    GOOG[column_name]=GOOG['Volume'].rolling(window= ma,center=False).mean()

GOOG[['Volume', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[56]:

# close prices for MSFT
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    MSFT[column_name]=MSFT['Close'].rolling(window= ma,center=False).mean()

MSFT[['Close', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[57]:

# Open prices for MSFT
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    MSFT[column_name]=MSFT['Open'].rolling(window= ma,center=False).mean()

MSFT[['Open', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[58]:

# trading volumes for MSFT
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    MSFT[column_name]=MSFT['Volume'].rolling(window= ma,center=False).mean()

MSFT[['Volume', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[59]:

# close prices for AMZN
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    AMZN[column_name]=AMZN['Close'].rolling(window= ma,center=False).mean()

AMZN[['Close', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[60]:

# Open prices for AMZN
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    AMZN[column_name]=AMZN['Open'].rolling(window= ma,center=False).mean()

AMZN[['Open', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# In[61]:

# trading volumes for AMZN
for ma in ma_day:
    column_name = 'Moving_Average for %s days'% (ma)
    AMZN[column_name]=AMZN['Volume'].rolling(window= ma,center=False).mean()

AMZN[['Volume', 'Moving_Average for 20 days', 'Moving_Average for 50 days', 'Moving_Average for 100 days']].plot(subplots = False, figsize =(10,4))


# ### Daily Return Analysis

# In[62]:

# after the above baseline analysis, let's analyze the risk of the stock. First, I will get the info on daily return of the stocks


# In[70]:

# for AAPL:
AAPL['Daily Return']=AAPL['Close'].pct_change()


# In[74]:

# plot it up:
AAPL['Daily Return'].plot(figsize=(10,4), legend=True, linestyle = '--', marker = 'o')


# In[75]:

# do the same thing for the other stocks: GOOG
GOOG['Daily Return']=GOOG['Close'].pct_change()
GOOG['Daily Return'].plot(figsize=(10,4), legend=True, linestyle = '--', marker = 'o')


# In[76]:

# do the same thing for the other stocks: MSFT
MSFT['Daily Return']=GOOG['Close'].pct_change()
MSFT['Daily Return'].plot(figsize=(10,4), legend=True, linestyle = '--', marker = 'o')


# In[77]:

# do the same thing for the other stocks: AMZN
AMZN['Daily Return']=AMZN['Close'].pct_change()
AMZN['Daily Return'].plot(figsize=(10,4), legend=True, linestyle = '--', marker = 'o')


# In[80]:

# do a histogram:
AAPL['Daily Return'].hist()


# In[81]:

GOOG['Daily Return'].hist()


# In[82]:

MSFT['Daily Return'].hist()


# In[83]:

AMZN['Daily Return'].hist()


# In[87]:

# do the kde and the histogram on the same figure
sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color = 'blue')


# In[89]:

# GOOG:
sns.distplot(GOOG['Daily Return'].dropna(), bins=100, color = 'purple')


# In[92]:

# MSFT
sns.distplot(MSFT['Daily Return'].dropna(), bins=100, color = 'green')


# In[93]:

# AMZN
sns.distplot(AMZN['Daily Return'].dropna(), bins = 100, color ='red')


# In[94]:

# Grab all the closing prices of the stocks to form a new dataframe
closing_df = web.DataReader(['AAPL','GOOG', 'MSFT', 'AMZN'], 'google', start, end)['Close']


# In[95]:

closing_df


# In[96]:

# calculate the percentage change for all
tech_pct = closing_df.pct_change()


# In[97]:

# check correlation between different stocks
# first, compare AAPL to itself to check if we get a linear relationship, if yes, move on
sns.jointplot('AAPL','AAPL', tech_pct, kind = 'scatter', color ='seagreen')


# In[98]:

# alright, it works. Let's compare between stocks
sns.jointplot('AAPL', 'GOOG', tech_pct, kind='scatter')


# In[99]:

# compare between AAPL and MSFT
sns.jointplot('AAPL', 'MSFT', tech_pct, kind='scatter')


# In[100]:

# compare between AAPL and AMZN
sns.jointplot('AAPL', 'AMZN', tech_pct, kind='scatter', color = 'red')


# In[101]:

# compare between GOOG and MSFT
sns.jointplot('GOOG', 'MSFT', tech_pct, kind='scatter')


# In[102]:

# compare between GOOG and AMZN
sns.jointplot('GOOG', 'AMZN', tech_pct, kind='scatter', color ='yellow')


# In[106]:

# compare between AMZN and MSFT
sns.jointplot('AMZN', 'MSFT', tech_pct, kind='scatter', color = 'green')


# In[109]:

# evaluating the Pearsonr values in the plots gives us a sense how correlated the different groups of data are.
# import the Pearsonr plot as a reference


# In[108]:

from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')


# In[112]:

# a very powerful tool in seaborn, pairplot shows the correlation comparision between all series in a dataframe
sns.pairplot(tech_pct.dropna())


# In[114]:

# use sns.PairGrid() for full control of the figure, including what kind of plots go in the diagonal, the upper triangle, 
# and the lower triangle


# In[117]:

returns_fig = sns.PairGrid(tech_pct.dropna())
# diagonal
returns_fig.map_diag(plt.hist, bins = 100)
# upper triangle
returns_fig.map_upper(plt.scatter, color = 'purple')
# lower triangle
returns_fig.map_lower(sns.kdeplot, cmap ='cool_d')


# In[118]:

# do the same analysis for closing_df 
returns_fig = sns.PairGrid(closing_df.dropna())
# diagonal
returns_fig.map_diag(plt.hist, bins = 100)
# upper triangle
returns_fig.map_upper(plt.scatter, color = 'green')
# lower triangle
returns_fig.map_lower(sns.kdeplot, cmap ='cool')


# In[121]:

# use the seaborn correlation matrix and heatmap to get the numerical values: (diagonal correlation matrix)
corrmat = tech_pct.dropna().corr()
sns.heatmap(corrmat, vmax=.8, square=True)


# In[ ]:

# it is seen that AMZN and GOOG has a strong relationship


# In[129]:

corrmat = tech_pct.dropna().corr()
sns.heatmap(corrmat, square=True)


# ### Risk Analysis

# In[143]:

# comparing the expected return with the standard deviation of the daily return
tech_clean = tech_pct.dropna()
area = np.pi*20

plt.scatter(tech_clean.mean(), tech_clean.std(), alpha = 0.5, s = area)

plt.xlim([0.0005,0.0025])
plt.ylim([0.005, 0.020])

plt.xlabel('Expected Returns')
plt.ylabel('Risk')

# Label the scatter plots
for label, x, y in zip(tech_clean.columns, tech_clean.mean(), tech_clean.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


# ### Value at Risk

# In[144]:

# value at risk (VaR): the amount of money one can expect to lose, a.k.a. putting at risk for a given confidence interval
sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color = 'blue')


# In[145]:

# use 0.05 empirial quantile (分位数):
tech_clean['AAPL'].quantile(0.05)


# In[146]:

# The 0.05 empirical quantile of daily returns is at -0.016. That means that with 95% confidence, 
# the worst daily loss will not exceed 1.6%. If we have a 1 million dollar investment, our one-day 5% VaR is
# 0.016 * 1,000,000 = $16,000.


# In[147]:

# do the same analysis for GOOG
sns.distplot(GOOG['Daily Return'].dropna(), bins=100, color = 'blue')


# In[148]:

# use 0.05 empirial quantile (分位数):
tech_clean['GOOG'].quantile(0.05)


# In[149]:

# The 0.05 empirical quantile of daily returns is at -0.014. That means that with 95% confidence, 
# the worst daily loss will not exceed 1.4%. If we have a 1 million dollar investment, our one-day 5% VaR is
# 0.014 * 1,000,000 = $14,000.


# In[150]:

# do the same analysis for MSFT
sns.distplot(MSFT['Daily Return'].dropna(), bins=100, color = 'blue')


# In[151]:

# use 0.05 empirial quantile (分位数):
tech_clean['MSFT'].quantile(0.05)


# In[152]:

# The 0.05 empirical quantile of daily returns is at -0.013. That means that with 95% confidence, 
# the worst daily loss will not exceed 1.3%. If we have a 1 million dollar investment, our one-day 5% VaR is
# 0.013 * 1,000,000 = $13,000.


# In[153]:

# do the same analysis for AMZN
sns.distplot(AMZN['Daily Return'].dropna(), bins=100, color = 'blue')


# In[154]:

# use 0.05 empirial quantile (分位数):
tech_clean['AMZN'].quantile(0.05)


# In[155]:

# The 0.05 empirical quantile of daily returns is at -0.016. That means that with 95% confidence, 
# the worst daily loss will not exceed 1.6%. If we have a 1 million dollar investment, our one-day 5% VaR is
# 0.016 * 1,000,000 = $16,000.


# In[156]:

# now try to calculate VaR using Monte Carlo method
# Monte Carlo analysis runs many trials with random market conditions, and then calculates portfolio losses for each trial. 
# After this, all these simulations are aggragated to establish how risky the stock is.


# In[157]:

# stock price = drift (predicted) + shock (random movement)


# In[158]:

# Use the geometric Brownian motion (GBM), which is technically known as a Markov process. 
# This means that the stock price follows a random walk and is consistent with (at the very least) the weak form of 
# the efficient market hypothesis (EMH): past price information is already incorporated and 
# the next price movement is "conditionally independent" of past price movements.


# In[159]:

# By simulating this series of steps of drift and shock thousands of times, 
# we can begin to do a simulation of where we might expect the stock price to be.


# In[160]:

# Set up the time horizon
days = 365

# thus the delta
dt = 1/days

# Now grab the mu (drift) from the expected return data we got for AAPL
mu = tech_clean.mean()['AAPL']

# Now grab the volatility of the stock from the std() of the average return
sigma = tech_clean.std()['AAPL']


# In[164]:

# create a function that takes in the starting price and number of days, and uses the sigma and mu
def stock_monte_carlo(start_price,days,mu,sigma):
    ''' This function takes in starting stock price, days of simulation, mu, sigma, and returns simulated price array'''
    
    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    
    # Define Shock and Drift arrays
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Run price array for number of days
    for x in range(1,days):
        
        # Calculate Shock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price


# In[162]:

# get the start price for AAPL: 97.32
AAPL.head()


# In[165]:

# run the random simulation for 100 times
start_price = 97.32

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Apple')


# In[166]:

# get the start price for GOOG: 718.27
GOOG.head()


# In[167]:

# run the random simulation for 100 times
start_price = 718.27

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Google')


# In[168]:

# get the start price for MSFT: 49.90
MSFT.head()


# In[169]:

# run the random simulation for 100 times
start_price = 49.90

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Microsoft')


# In[170]:

# get the start price for AMZN: 712.33
AMZN.head()


# In[171]:

# run the random simulation for 100 times
start_price = 712.33

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Amazon')


# In[172]:

# get the end results (at day = 365) for much larger runs

# Set a large numebr of runs
runs = 10000

start_price = 97.32

# Create an empty matrix to hold the end price data
simulations = np.zeros(runs)

# Set the print options of numpy to only display 0-5 points from an array to suppress output
np.set_printoptions(threshold=5)

for run in range(runs):    
    # Set the simulation data point as the last stock price for that run
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];


# In[175]:

# Now define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Now plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=2, color='g')

# Title
plt.title("Final price distribution for Apple Stock after %s days" % days, weight='bold');


# In[174]:

# the 1% empirical quantile of the final price distribution to estimate the Value at Risk for the Apple stock, 
# which looks to be $2.25 for every investment of 97.32 (the price of one inital Apple stock).
# This basically menas for every initial stock you purchase your putting about $2.25 at risk 99% of the time 
# from our Monte Carlo Simulation.


# In[176]:

-


# In[177]:

# the 1% empirical quantile of the final price distribution to estimate the Value at Risk for the Google stock, 
# which looks to be $16.67 for every investment of 718.27 (the price of one inital Google stock).
# This basically menas for every initial stock you purchase your putting about $16.67 at risk 99% of the time 
# from our Monte Carlo Simulation.


# In[178]:

# Now get the results for Microsoft

# Set a large numebr of runs
runs = 10000

start_price = 49.90

# Create an empty matrix to hold the end price data
simulations = np.zeros(runs)

# Set the print options of numpy to only display 0-5 points from an array to suppress output
np.set_printoptions(threshold=5)

for run in range(runs):    
    # Set the simulation data point as the last stock price for that run
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];
    
# Now define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Now plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=2, color='g')

# Title
plt.title("Final price distribution for Microsoft Stock after %s days" % days, weight='bold');


# In[179]:

# the 1% empirical quantile of the final price distribution to estimate the Value at Risk for the Google stock, 
# which looks to be $1.16 for every investment of 49.90 (the price of one inital Microsoft stock).
# This basically menas for every initial stock you purchase your putting about $1.16 at risk 99% of the time 
# from our Monte Carlo Simulation.


# In[180]:

# Now get the results for Amazon

# Set a large numebr of runs
runs = 10000

start_price = 712.33

# Create an empty matrix to hold the end price data
simulations = np.zeros(runs)

# Set the print options of numpy to only display 0-5 points from an array to suppress output
np.set_printoptions(threshold=5)

for run in range(runs):    
    # Set the simulation data point as the last stock price for that run
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];
    
# Now define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Now plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=2, color='g')

# Title
plt.title("Final price distribution for Amazon Stock after %s days" % days, weight='bold');


# In[181]:

# the 1% empirical quantile of the final price distribution to estimate the Value at Risk for the Amazon stock, 
# which looks to be $16.42 for every investment of 49.90 (the price of one inital Amazon stock).
# This basically menas for every initial stock you purchase your putting about $16.42 at risk 99% of the time 
# from our Monte Carlo Simulation.


# In[ ]:



