#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


# import data
df = pd.read_csv('ab_data.csv')

# show top rows
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[3]:


df_length = len(df)         
print(df_length)


# c. The number of unique users in the dataset.

# In[4]:


len(df.user_id.unique())


# d. The proportion of users converted.

# In[5]:


df.converted.sum()/df_length


# e. The number of times the `new_page` and `treatment` don't line up.

# In[6]:


# Looking for rows where treatment/control doesn't match with old/new pages respectively
df_t_not_n = df[(df['group'] == 'treatment') & (df['landing_page'] == 'old_page')]
df_not_t_n = df[(df['group'] == 'control') & (df['landing_page'] == 'new_page')]

# Adding lengths of mismatches
mismatch= len(df_t_not_n) + len(df_not_t_n)

# Create one dataframe from it
mismatch_df = pd.concat([df_t_not_n, df_not_t_n])

print (mismatch)


# f. Do any of the rows have missing values?

# In[7]:


df.isnull().values.any()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


# New dataframe
df2 = df

# Remove mismatching rows
mismatch_index = mismatch_df.index
df2 = df2.drop(mismatch_index)


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[10]:


# Finding unique users
print("Unique users:", len(df2.user_id.unique()))

# Checking for not unique users
print("Non-unique users:", len(df2)-len(df2.user_id.unique()))


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


# Finding duplicate user
df2[df2.duplicated('user_id')]


# c. What is the row information for the repeat **user_id**? 

# In[12]:


# Finding duplicate row number under user ids
df2[df2['user_id']==773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


# Drop duplicate row
df2.drop(labels=1899, axis=0, inplace=True)


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[14]:


# Probability of user converting
print("Probability of a user converting:", df2.converted.mean())


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


# Probability of control group users converting
print("Probability of control group converting:", 
      df2[df2['group']=='control']['converted'].mean())


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


# Probability of treatment group users converting
print("Probability of treatment group converting:", 
      df2[df2['group']=='treatment']['converted'].mean())


# d. What is the probability that an individual received the new page?

# In[17]:


# Probability of a user recieved new page
print("Probability an user recieving new page:", 
      df2['landing_page'].value_counts()[0]/len(df2))


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# According to the probabilities, the control group (the group with the old page) converted at a higher rate than the teatment (the group with the new page). The magnitude of this change is very small with a difference of roughly 0.2%.
# 
# Given the data in Question 4 so far, the probability that an individual recieved a new page is roughly 0.5, this means that it is not possible for there to be a difference in conversion based on being given more opportunities to do so. For instance, if the probability of recieving a new page was higher relative to the old page then it would be observed that the rate of conversion would naturally increase.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# Our null hypothesis is that **$H_{0} : $** **$p_{new}$** - *$p_{old}$* <= 0 
# 
# That means that the null hypothesis is that the difference between the population conversion rate of users given the new page and the old page will be the same or lower than zero which means that the old page has a higher population conversion rate.

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# According to the assumption, **$p_{new}$** = *$p_{old}$* .So we should calculate the average of the real $p_{new}$ and $p_{old}$ (probability of conversion given new page and old page respectively) to calculate $p_{mean}$.

# In[18]:


# Calculating probability of conversion for new page
p_new = df2[df2['landing_page']=='new_page']['converted'].mean()

print("Probability of conversion for new page (p_new):", p_new)


# In[19]:


# Calculating probability of conversion for old page
p_old = df2[df2['landing_page']=='old_page']['converted'].mean()

print("Probability of conversion for old page (p_old):", p_old)


# In[20]:


# Then we take the mean of these two probabilities
p_mean = np.mean([p_new, p_old])

print("Probability of conversion under null hypothesis (p_mean):", p_mean)


# In[21]:



# We calculate the difference in probability of conversion for new and old page (not under H_0)
p_diff = p_new-p_old

print("Difference in probability of conversion for new and old page (not under null hypothesis):", p_diff)


# So we have:
# 
# $p_{new}: 0.1188$
# 
# $p_{old}: 0.1203$
# 
# Pmean = Pnew0 = Pold0 = 0.1196

# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# Pmean = Pnew0 = Pold0 = 0.1196

# c. What is $n_{new}$?

# In[29]:


# Calculate n_new and n_old
n_new, n_old = df2['landing_page'].value_counts()

print("new:", n_new)


# d. What is $n_{old}$?

# In[30]:


print("old:", n_old)


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[24]:


# Simulating conversion rates under null hypothesis
new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_mean, (1-p_mean)])


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[25]:


# Simulating conversion rates under null hypothesis
old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_mean, (1-p_mean)])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[26]:


new_page_converted.mean()-old_page_converted.mean()


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[27]:


p_diffs = []

# Re-run simulation 10,000 times
# trange creates an estimate for how long this program will take to run
for i in range(10000):
    new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_mean, (1-p_mean)])
    old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_mean, (1-p_mean)])
    p_diff = new_page_converted.mean()-old_page_converted.mean()
    p_diffs.append(p_diff)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[28]:


# Ploting histogram
plt.hist(p_diffs, bins=25)
plt.title('Simulated Difference of New Page and Old Page Converted Under the Null')
plt.xlabel('Page difference')
plt.ylabel('Frequency')
plt.axvline(x=(p_new-p_old), color='r', linestyle='dashed', linewidth=1, label="Real difference")
plt.axvline(x=(np.array(p_diffs).mean()), color='g', linestyle='dashed', linewidth=1, label="Simulated difference")
plt.legend()
plt.show()


# The simulated data creates a normal distribution as we expected due to how the data was generated. The mean of this normal distribution is 0, which  is also expected under the null hypothesis.

# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[31]:


#Difining p_diff
p_diff = p_new - p_old
#Finding proportion of p_diffs greater than the actual difference
greater_than_diff = [i for i in p_diffs if i > p_diff]

#Calculating and printing values
print("Actual difference:" , p_diff)

p_greater_than_diff = len(greater_than_diff)/len(p_diffs)

print('Proportion greater than actual difference:', p_greater_than_diff)

print('As a percentage: {}%'.format(p_greater_than_diff*100))


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# 
# If our sample conformed to the null hypothesis then we'd expect the proportion greater than the actual difference to be 0.5. However, we calculate that almost 90% of the population in our simulated sample is above the real difference which does not only suggest that the new page does not do significantly better than the old page, but actually performs much worse.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[32]:


import statsmodels.api as sm

# Calculating number of conversions
# Some of these values were defined ealier in this notebook: n_old and n_new

convert_old = len(df2[(df2['landing_page']=='old_page')&(df2['converted']==1)])
convert_new = len(df2[(df2['landing_page']=='new_page')&(df2['converted']==1)])

print("convert_old:", convert_old, 
      "\nconvert_new:", convert_new,
      "\nn_old:", n_old,
      "\nn_new:", n_new)


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[33]:


# Finding z-score and p-value
z_score, p_value = sm.stats.proportions_ztest(count=[convert_new, convert_old], 
                                              nobs=[n_new, n_old], alternative ='larger')
print("z-score:", z_score,
     "\np-value:", p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# z-score is the number of standard deviations from the mean a data point is. But more technically it’s a measure of how many standard deviations below or above the population mean a raw score is.
# 
# The differences between the lines shown in the histogram above is -1.31 standard deviations. The p-value is roughly 90.0% which is the probability that this result is due to random chance, which is too high to reject the null hypothesis and thus we fail to do so.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# We will use a logistic regression.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[34]:


df3 = df2 # Copying dataframe in case of a mistake

df3['intercept'] = pd.Series(np.zeros(len(df3)), index=df3.index)
df3['ab_page'] = pd.Series(np.zeros(len(df3)), index=df3.index)

# Finding indexes that need to be changed for treatment group
index_to_change = df3[df3['group']=='treatment'].index

# Changing values
df3.set_value(index=index_to_change, col='ab_page', value=1)
df3.set_value(index=df3.index, col='intercept', value=1)

# Changing datatype
df3[['intercept', 'ab_page']] = df3[['intercept', 'ab_page']].astype(int)

# Moving "converted"
df3 = df3[['user_id', 'timestamp', 'group', 'landing_page', 'ab_page', 'intercept', 'converted']]

df3[df3['group']=='treatment'].head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[35]:


# Setting up logistic regression
logit = sm.Logit(df3['converted'], df3[['ab_page', 'intercept']])

# Calculating results
result=logit.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[36]:


result.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>

# The p-value is 0.19 which is very different to the one that we calculated in Part II. The reason behind this is because in this case we do a 2 tailed test.
# 
# However, the p-value is again too high to reject the null hypothesis.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# Considering other factors is a good idea since it might give us more insight and different perspective to the analysis. 
# 
# However, adding too many factors into the analysis can lead to false outcome of the analysis.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns 

# In[38]:


# Importing data
df_countries = pd.read_csv('countries.csv')

df_countries.head()


# In[39]:


# Creating dummy variables
df_dummy = pd.get_dummies(data=df_countries, columns=['country'])

# Performing merge and storing into a new df
df4 = df_dummy.merge(df3, on='user_id')

# Sorting columns
df4 = df4[['user_id', 'timestamp', 'group', 'landing_page', 
           'ab_page', 'country_CA', 'country_UK', 'country_US',
           'intercept', 'converted']]

# Fix Data Types
df4[['ab_page', 'country_CA', 'country_UK', 'country_US','intercept', 'converted']] =df4[['ab_page', 'country_CA', 'country_UK', 'country_US','intercept', 'converted']].astype(int)

df4.head()


# In[40]:


# Creating logit_countries object
logit_countries = sm.Logit(df4['converted'], 
                           df4[['country_UK', 'country_US', 'intercept']])

# Fiting
result2 = logit_countries.fit()


# In[41]:


# Show results
result2.summary()


# It seems that the split of the countries had some impact on the conversion rate, however, the p value is still too high in order for us to reject the null hypothesis.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[46]:


# Creating logit_countries object
df4['US_ind_ab_page'] = df4['country_US']*df4['ab_page']
df4['CA_ind_ab_page'] = df4['country_CA']*df4['ab_page']
logit_countries2 = sm.Logit(df4['converted'], df4[['intercept', 'ab_page', 'country_US', 'country_CA', 'US_ind_ab_page', 'CA_ind_ab_page']])

# Fitting
result3 = logit_countries2.fit()


# In[47]:


# Showing final results
result3.summary2()


# When adding everything, the p values increases moderately so there are still not enough evidence to reject the null hypothesis.

# <a id='conclusions'></a>
# ## Finishing Up
# 
# Although it would seem initially that there is a difference between the conversion rates of new and old pages, we determined there is just not enough evidence to reject the null hypothesis. From the histogram shown in this report, it seems that the new page does worse than the old one.
# 
# It was also found that this was not dependent on countries with conversion rates being roughly the same in the UK as in the US. The test conditions were fairly good as well, users had a roughly 50% chance to recieve the new and old pages and the sample size of the initial dataframe is sufficiently big such that collecting data is likely not a good use of resources.
# 
# I would recommend to the e-commerce company to keep using the old version of the website as from the data gathered and the analysis, we determined that the old version performs better than the new one.
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[48]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




