---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python [conda env:comsocsci2022]
    language: python
    name: conda-env-comsocsci2022-py
---

# Part 2: activity on Reddit and GME prices


## Part 2.1 : Plotting prices and comments using line-graphs.

```python
import matplotlib.pylab as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy import stats
```

```python
#Setting up data
GME_marketdata = pd.read_csv('../data/GME.csv', parse_dates=['Date']).set_index('Date')
rolled_marketdata = GME_marketdata['Volume'].rolling('7D', center=True).mean()

GME_comments = pd.read_csv('../data/wallstreetbet_gamestop_comments_cleaned.csv')
GME_comments['date_of_creation'] = pd.to_datetime(GME_comments['date_of_creation'], unit='s')
GME_comments = GME_comments.sort_values('date_of_creation').set_index('date_of_creation')
GME_comments = GME_comments.loc['2020-1-1':'2021-1-25']
GME_comments['daily'] = GME_comments.rolling('1D', center=True).count()['id']
GME_comments['weekly'] = GME_comments['daily'].rolling('7D', center=True).mean()
```

```python
def setup_mpl():
    mpl.rcParams['font.family'] = 'Helvetica Neue'
    mpl.rcParams['lines.linewidth'] = 1
setup_mpl()
myFormat = mpl.dates.DateFormatter('%b %Y')
```

### 1. Plot the daily volume of the GME stock over time using the GME market data. On top of the daily data, plot the rolling average, using a 7 days window (you can use the function pd.rolling). Use a log-scale on the y-axis.

```python
fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi=400)
ax.plot(GME_marketdata.index,GME_marketdata.Volume, ls = '--', alpha = 0.5, label="Daily volume")
ax.plot(rolled_marketdata.index,rolled_marketdata.values, color='black', label="1 week rolling average")
ax.set_ylabel('Volume (USD)')
ax.set_yscale('log')
ax.legend()
ax.xaxis.set_major_formatter(myFormat)
plt.title("Figure 1: Daily GME stock volume ", y=-0.15, fontsize=8)
plt.show()
```

### 2. Now make a second plot where you plot the total number of comments on Reddit per day. Follow the same steps you followed in step 1.

```python
fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi=400)
ax.plot(GME_comments['daily'].index,GME_comments['daily'].values, ls = '--', alpha = 0.5, label="Daily comments")
ax.plot(GME_comments['weekly'].index,GME_comments['weekly'].values, color='black', label="1 week rolling average")
ax.set_ylabel('Comments')
ax.set_yscale('log')
ax.legend()
ax.xaxis.set_major_formatter(myFormat)
plt.title("Figure 2: Daily number of comments on r/wallstreetbet subreddit", y=-0.15, fontsize=8)
plt.show()
```

### 3. What is the advantage of using the log-scale on the y-axis? What is the advantage of using a rolling-window?


Using a log scale on the y-axis allows us to see long term trends, ranging from very few comments in the start and the multiple orders of comments in the end of the time period. If we had used cartesian coordinates the trend of the small numbers would not be visible.
The advantage of using a rolling window, here centered ont he period, smoothes the data in order to capture the overall trend of the data.


### 4. Now take a minute to look at these two figures. Then write in a couple of lines: What are the three most important observations you can draw by looking at the figures?

```python
fig, ax = plt.subplots(figsize=(10,5), dpi=400)
ax.plot(rolled_marketdata.index,rolled_marketdata.values, color='black', label="1 week rolling average volume")
ax.set_ylabel('Volume (USD)')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(myFormat)
ax2 = ax.twinx()
ax2.plot(GME_comments['weekly'].index,GME_comments['weekly'].values, color='red', label="1 week rolling average comments")
ax2.set_ylabel('Comments')
ax2.set_yscale('log')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
fig.tight_layout()
plt.savefig('GME weekly volume and comments.png')
plt.title("Figure 3: Rolling weekly GME stock volume and number of comments on r/wallstreetbet subreddit", y=-0.15, fontsize=10)
plt.show()
```

!!!! Skal vi nævne at der er tale om ln(x) ?
!!!! Calc stats for these claims, eg Pearson ?

Figure 3 shows the rolling weekly GME stock volume and number of comments in a dual axis chart, showing a clear correlation between volume and number of comments. 

2)

3)


## Part 2.2: Returns vs number of comments using scatter-plots. In this part of the assignment, we will look at the association between GME market indicators and the volume of comments on Reddit


### 1. & 2. Compute the daily log-returns and the daily log-change in number of new comments

```python
log_return_comments = pd.DataFrame(np.log(GME_marketdata['Close'] / GME_marketdata['Close'].shift(1)))
daily_comments = GME_comments.resample('1D').count()['id']
daily_comments = daily_comments[daily_comments!=0]
daily_log_change = np.log(daily_comments / daily_comments.shift(1))
log_return_comments['comments'] = daily_log_change
log_return_comments.dropna(inplace=True)
```

### 3. Compute the correlation coefficient between the series computed in step 1 and step 2. Is the correlation statistically significant?

```python
print(f" Pearson correlation coeffcient for the daily log change in comments and return:{stats.pearsonr(log_return_comments['Close'],log_return_comments['comments'])[0]:.2f}, p-value:{stats.pearsonr(log_return_comments['Close'],log_return_comments['comments'])[1]:.2f}")
```

The Pearson correlation coefficient shows a weak positive linear relationship between daily log change and return, but the p-value = 0.00 suggests that it is statistically significant.


### 4. Make a scatterplot of the daily log-return on investment for the GME stock against the daily log-change in number of comments.

```python
log_return_comments['close_price'] = GME_marketdata['Close']
log_return_comments['color'] = 'blue'
log_return_comments['color'].loc['2021'] = 'green'

fig, ax = plt.subplots(dpi=400)
ax.scatter(log_return_comments['Close'],log_return_comments['comments'], s=log_return_comments['close_price'], c=log_return_comments['color'], alpha=0.7)
ax.set_ylabel('Daily log-change in number of comments')
ax.set_xlabel('Daily log-return on investment for the GME stock')

for size in [2, 10, 50]:
    plt.scatter([], [], c='k', alpha=0.3, s=size, label=str(size) +' USD')

legend1 = plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Close price')
legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color='w', label='2020', markerfacecolor='b', markersize=10),
                   mpl.lines.Line2D([0], [0], marker='o', color='w', label='2021', markerfacecolor='g', markersize=10)
                   ]
ax.legend(handles=legend_elements)
ax.add_artist(legend1)
plt.title("Figure 4", y=-0.22, fontsize=6)
plt.show()
```

### 5. Now take a minute to look at the figure you just prepared. Then write in a couple of lines: What are the three most salient observations you can draw by looking at it? 


!!!!!!Skal vi lave linear regression på 2021 data?

When looking at Figure 4 we see an overall correlating trend between the daily log-change in comments and log-return on  investment. This trend is even more pronounced for the 2021 data where we observe a linear relationship which suggests a power law relationship, log-change in comments change by an order of magnitude compared to the daily log-return, equal to the slope of the fitted line.


### 6. Based on the exploratory data visualization in Exercises 2 and 3, what can you conclude on the research question: *Is the activity on wallstreetbet related to the price of the GME stock?*


There is evidence causality between the activity on wallstreetbet and the price of the GME stock, but there is not enough evidence to conclude the direction. The activity on reddit is certainly to a degreee a result of market changes for the GME stock, but the reverse could also be the case.
