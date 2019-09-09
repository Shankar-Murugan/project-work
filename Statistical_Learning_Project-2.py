#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np


# In[43]:


res1 = pd.read_csv('responses.csv')


# In[44]:


res1.shape


# In[45]:


res1.columns


# In[46]:


for col in res1.columns:
        print(col)


# In[47]:


res1.info()


# In[48]:


res1.shape


# In[49]:


res2 = res1[['Healthy eating','Finances','Gender','Village - town']]
res2.head()


# In[56]:


res2.isnull().sum()


# In[57]:


res2.mean()
res2=res2.fillna(res2.mean())
res2.head()


# In[58]:


res2.isnull().sum()


# In[59]:


res2.loc[:,"Gender"].mode()


# In[64]:


res2["Gender"].fillna("female", inplace = True)


# In[65]:


res2.loc[:,"Village - town"].mode()


# In[66]:


res2["Village - town"].fillna("city", inplace = True) 


# In[67]:


res2.isnull().sum()


# In[68]:


res2.info()


# In[50]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot(x='Gender',data=res2,hue='Healthy eating')


# In[51]:


sns.countplot(x='Gender',data=res2,hue='Finances')


# In[52]:


sns.countplot(x='Village - town',data=res2,hue='Healthy eating')


# In[53]:


sns.countplot(x='Village - town',data=res2,hue='Finances')


# In[60]:


sns.boxplot(x="Gender", y="Healthy eating", data=res2,palette='rainbow')


# #### From the above box plot, we can see a low out layer in the female Healthy eating habit survey.

# In[61]:


sns.boxplot(x="Village - town", y="Healthy eating", data=res2,palette='rainbow')


# #### From the above box plot, we can see a low outlier in the both village and city Healthy eating habit survey.

# In[62]:


sns.boxplot(x="Gender", y="Finances", data=res2,palette='rainbow')


# #### From the above box plot, There is no outliers in the both feamle and male finace survey.

# In[63]:


sns.boxplot(x="Village - town", y="Finances", data=res2,palette='rainbow')


# #### From the above box plot, There is no outliers in the both feamle and male finace survey.

# In[54]:


sns.boxplot(x="Village - town", y="Finances",hue='Gender',data=res2,palette='rainbow')


# From the above box plot, we concluded that village females mean more than others. Others are more or less equal distribution.

# In[55]:


sns.boxplot(x="Village - town", y="Healthy eating",hue='Gender',data=res2,palette='rainbow')


# From the above box plot, we can see few higher and lower out liers in the village female healthy eating.
# And a outlier can see in the city female healthy eating. And ther is no out liers in the male's healthy eating.

# In[127]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(res2, hue = 'Gender')


# From the above plot we can see females have more helthier eating habit than male.

# In[128]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(res2, hue = 'Place')


# From the above plot we can see city people saving more money than village people.

# In[95]:


sns.stripplot(x="Finances", y="Place", data=res2,jitter=True,hue='Gender',palette='Set1')


# In[96]:


sns.stripplot(x="Place", y="Healthy eating", data=res2,jitter=True,hue='Gender',palette='Set1')


# In[69]:


res2.groupby(["Gender"]).count()


# In[130]:


res2.groupby(["Place"]).count()


# ### Insite:
# 1. Over all Female better than male in Finances and Healthy eating.
# 2. Over all city people better than village people in Finance and Healthy eating.
# 3. Females in villages are the best comparing to others in the finance survey.
# 4. Females in cities are the best comparing to others in the Healty eating survey.

# ## Conclusion:
#     
# Is saving money (finances) gender dependant?
# 
# Not sure, because mostly means of the both male and female are more or less equal.
# 
# 
# Is there any differences in money saving (finances) between people from city or village?
# 
# Yes, over all city people better than village people
# 
# 
# Do Women have a Healthier Lifestyle than Men?
# 
# Yes women have a Heealthier Lifestyle than man.
# 
# 
# Are Village People Healthier than City People?
# 
# No, Village people not healthier than city people.
# 
# 
# Is saving money (finances) gender dependant?
# Is there any differences in money saving (finances) between people from city or village?
# Do Women have a Healthier Lifestyle than Men?
# Are Village People Healthier than City People?

# # Categorical variable encoding

# In[86]:


res2['Gender'] = res1.Gender.astype('category')
res2.dtypes


# In[87]:


res2 = res2.rename(columns = {"Village - town": "Place"})
res2.info()


# In[88]:


res2['Place'] = res2.Place.astype('category')
res2.dtypes


# In[97]:


x = res2["Gender"]


# In[98]:


x = pd.get_dummies(x, drop_first=True)


# In[99]:


x.head()


# In[100]:


y = res2["Place"]


# In[101]:


y = pd.get_dummies(y, drop_first=True)


# In[102]:


y.head()


# In[103]:


res3 = pd.concat([res2,pd.get_dummies(res2['Gender'], drop_first=True)],axis=1)


# In[104]:


res3.head()


# In[105]:


res3 = res3.rename(columns = {"male": "Gend"})


# In[106]:


res3.head()


# In[107]:


res4 = res3.drop(labels= "Gender" , axis = 1)
res4.head()


# In[108]:


res5 = pd.concat([res4,pd.get_dummies(res4['Place'], drop_first=True)],axis=1)


# In[109]:


res5.head()


# In[110]:


res6 = res5.drop(labels= "Place" , axis = 1)
res6.head()


# In[111]:


res7 = res6.rename(columns = {"Gend": "Gender"})


# In[112]:


res8 = res7.rename(columns = {"village": "Place"})


# In[113]:


res8.head()


# # Hypothesis Statement: Are Village People Healthier than City People?

# H0 ---->  There is no significant difference between both means. So accept null hypotheses. The Statement is not valied.
# Ha ---->  There is a significant difference between both means. So accept alternate hypotheses. The Statement is valied.

# In[114]:


from    scipy.stats             import  ttest_1samp,ttest_ind, wilcoxon, ttest_ind_from_stats
import  scipy.stats             as      stats  
from    statsmodels.stats.power import  ttest_power
import  matplotlib.pyplot       as      plt


# In[116]:


city_h = res8[res8['Place']==0]['Healthy eating']
village_h = res8[res8['Place']==1]['Healthy eating']
t_statistic, p_value  =  stats.ttest_ind(city_h,village_h)
print('P Value %1.3f' % p_value)


# Since P value is not less than 0.05, we accept the null hypotheses and there is no significant difference between city and village people healthier eating habit.

# ### As per two sample t test, There is no sinificant difference between mean and p vaue not less than 0.05, so there is no differences in money saving (finances) between people from city or village.

# # Hypothesis Statement: Is there any differences in money saving (finances) between people from city or village?

# H0 ---->  There is no significant difference between both means. So accept null hypotheses. The Statement is not valied.
# Ha ---->  There is a significant difference between both means. So accept alternate hypotheses. The Statement is valied.

# In[117]:


city_f = res8[res8['Place']==0]['Finances']
village_f = res8[res8['Place']==1]['Finances']
t_statistic, p_value  =  stats.ttest_ind(city_f,village_f)
print('P Value %1.3f' % p_value)


# Since P value is less than 0.05, we reject the null hypotheses and accept the alternate hypothesis and ensure that significant difference between city and village people money saving habits.

# ### As per two sample t test, There is a sinificant difference between mean and p vaue less than 0.05, so there is some differences in money saving (finances) between people from city or village.

# # Hypotheses Statement: Do Women have a Healthier Lifestyle than Men?

# H0 ---->  There is no significant difference between both means. So accept null hypotheses. The Statement is not valied.
# Ha ---->  There is a significant difference between both means. So accept alternate hypotheses. The Statement is valied.

# In[121]:


female_h = res8[res8['Gender']==0]['Healthy eating']
male_h = res8[res8['Gender']==1]['Healthy eating']
t_statistic, p_value  =  stats.ttest_ind(female_h,male_h)
print('P Value %1.3f' % p_value)


# Since P value is less than 0.05, we reject the null hypotheses and accept the alternate hypothesis and ensure that significant difference between male and female healtheir eating habit.

# H0 ---->  There is no significant difference between both means. So accept null hypotheses. The Statement is not valied.
# Ha ---->  There is a significant difference between both means. So accept alternate hypotheses. The Statement is valied.

# ### As per two sample t test, There is a sinificant difference between mean and p value less than 0.05, so we accept the following statement "Women have a Healthier Lifestyle than Men". 

# # Hypothesis Statement: Is saving money (finances) gender dependant?
# 

# In[123]:


female_f = res8[res8['Gender']==0]['Finances']
male_f = res8[res8['Gender']==1]['Finances']
t_statistic, p_value  =  stats.ttest_ind(female_f,male_f)
print('P Value %1.3f' % p_value)


# Since P value is not less than 0.05 so we accept the null hypotheses and reject the alternate hypothesis and ensure that there is no significant difference between male and female money saving eating habit.

# ### As per two sample t test, There is no sinificant difference between mean and p value is not less than 0.05, so saving money (finances) is not gender dependant. 

# # Conclusion: As per our two sample t test hypothsis analysis we have concluded that City people saving more money than village people and woman have helthier eating habit than man.
