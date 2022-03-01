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

<!-- #region -->
## Link to Git repository
https://github.com/realnikolaj/comsocsci2022group6



## Contribution statement

!!!!!!!!!!!!!!!Husk contribution statement her!!!!!!!!!!!!!!



<!-- #endregion -->

# Part 1: Datasets for Computational Social Science


### Big. How large is this data (approximately)? Could you collect the same amount of information via surveys?


We have 14715 submissions by 9375 unique authors and 644,397 comments on those submissions. It would probabaly be costly to obtain the same amount of information by surveys.


### Always-on. Can you keep collecting data over time?


The data is always-on, at least submission and comment are saved in https://files.pushshift.io, while scores will not be correctly calculated but was not here.


### Non-reactive. Is the dataset non-reactive?


The data is non-reactive, as people are probably not aware that data is used for experiments like these.


### Incomplete. Do you think the dataset captures entirely the unfolding of events leading to the GME stock rise in price?


The dataset does not capture all events leading to the GME stock rise. There are probably other online platforms like telegram and also the search criterion for reddit information may not capture all relevant data. Other than that market commentators, stock analysts and word of mouth are also sources of information exchange regarding the matter.


### Inaccessible. Is the data accessible? 


Reddit data is publicly accessible from PushShift API.


### Non-representative. Do you think that the conclusions we will draw by analyzing this dataset are specific to the GME events? Or could they instead help us understand social phenomena more in general? If yes, which phenomena could you think of? Of not, what are the aspects that make this dataset non-representative?


The conclusions will not be specific to the GME events, they could also tell us about the phenomena of manipulating stock prices in groups.


### Drifting. Is there any source of drift in this dataset (within the period observed)? 


Public perception of the phenomena, e.g. conveyed thorugh public news, will change the influx of users in this subreddit and investors.


### Algorithmically confounded. Is the dataset algorithmically confounded? If yes, why?


The score system will influence the type of posts, as thought provoking posts will probably get a score and comments. Some users will try to post/comment from a principle to improve ones score.


### Dirty. What aspect may make this dataset dirty?


As everyone can post/comment, there are unwanted data such as deleted comments and users, plus junk and spam.


### Sensitive. Is there any sensitive information in the data?


Usernames would not be considered sensitive, but could potentially be combined with other information, such as timestamps and crosreferencing other sites to identify the person.
