#!/usr/bin/env python
# coding: utf-8

# # K nearest neighbors

# KNN falls in the supervised learning family of algorithms. Informally, this means that we are given a labelled dataset consiting of training observations (x, y) and would like to capture the relationship between x and y. More formally, our goal is to learn a function h: X→Y so that given an unseen observation x, h(x) can confidently predict the corresponding output y.
# 
# In this module we will explore the inner workings of KNN, choosing the optimal K values and using KNN from scikit-learn.

# ## Overview
# 
# 1. Read the problem statement.
# 
# 2. Get the dataset.
# 
# 3. Explore the dataset.
# 
# 4. Pre-processing of dataset.
# 
# 5. Visualization
# 
# 6. Transform the dataset for building machine learning model.
# 
# 7. Split data into train, test set.
# 
# 8. Build Model.
# 
# 9. Apply the model.
# 
# 10. Evaluate the model.
# 
# 11. Finding Optimal K value
# 
# 12. Repeat 7, 8, 9 steps.

# ### Dataset
# 
# The data set we’ll be using is the Iris Flower Dataset which was first introduced in 1936 by the famous statistician Ronald Fisher and consists of 50 observations from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals.
# 
# **Download the dataset here:**
# - https://www.kaggle.com/uciml/iris
# 
# **Train the KNN algorithm to be able to distinguish the species from one another given the measurements of the 4 features.**

# ## Load data

# ### Question 1
# 
# Import the data set and print 10 random rows from the data set
# 
# Hint: use **sample()** function to get random rows

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


D1=pd.read_csv('iris.csv')


# In[3]:


D1.sample(10)


# ## Data Pre-processing

# ### Question 2 - Estimating missing values
# 
# Its not good to remove the records having missing values all the time. We may end up loosing some data points. So, we will have to see how to replace those missing values with some estimated values (median)
# Calculate the number of missing values per column
- don't use loops
# In[4]:


D1.info()


# In[5]:


D1.isnull().sum()


# Fill missing values with median of that particular column

# In[6]:


D1.mean()
D1=D1.fillna(D1.mean())
D1.head()


# In[ ]:





# ### Question 3 - Dealing with categorical data
# 
# Change all the classes to numericals (0 to 2)
# 
# Hint: use **LabelEncoder()**

# In[7]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
D1['Species']= label_encoder.fit_transform(D1['Species'])
D1.head()


# ### Question 4
# 
# Observe the association of each independent variable with target variable and drop variables from feature set having correlation in range -0.1 to 0.1 with target variable.
# 
# Hint: use **corr()**

# In[8]:


cor = D1.corr()
cor


# In[9]:


D1=D1.drop('Id',axis=1)


# In[10]:


D1.head()


# ### Question 5
# 
# Observe the independent variables variance and drop such variables having no variance or almost zero variance (variance < 0.1). They will be having almost no influence on the classification
# 
# Hint: use **var()**

# In[11]:


D1.var()


# In[ ]:





# ### Question 6
# 
# Plot the scatter matrix for all the variables.
# 
# Hint: use **pandas.plotting.scatter_matrix()**
# 
# you can also use pairplot()

# In[12]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(D1, hue = 'Species')


# In[ ]:





# ## Split the dataset into training and test sets
# 

# ### Question 7
# 
# Split the dataset into training and test sets with 80-20 ratio
# 
# Hint: use **train_test_split()**

# In[13]:


from sklearn.model_selection import train_test_split
y = D1[['Species']]
X = D1.drop(columns=['Species'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:





# ## Build Model

# ### Question 8
# 
# Build the model and train and test on training and test sets respectively using **scikit-learn**.
# 
# Print the Accuracy of the model with different values of **k = 3, 5, 9**
# 
# Hint: For accuracy you can check **accuracy_score()** in scikit-learn

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# ## Find optimal value of K

# ### Question 9 - Finding Optimal value of k
# 
# - Run the KNN with no of neighbours to be 1, 3, 5 ... 19
# - Find the **optimal number of neighbours** from the above list

# In[19]:


import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[29]:


acc_score=[]

for k in range (1,20,2):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    score=accuracy_score(y_test, y_predict)
    acc_score.append(score)
print("accuracy score and K values {}".format(max(acc_score)))


# In[ ]:





# ## Plot accuracy

# ### Question 10
# 
# Plot accuracy score vs k (with k value on X-axis) using matplotlib.

# In[30]:


import matplotlib.pyplot as plt 
plt.plot(acc_score)


# In[ ]:





# # Breast cancer dataset - OPTIONAL

# ## Read data

# ### Question 1
# Read the data given in bc2.csv file

# In[ ]:





# ## Data preprocessing

# ### Question 2
# Observe the no.of records in dataset and type of each column

# In[ ]:





# In[ ]:





# ### Question 3
# Use summary statistics to check if missing values, outlier and encoding treament is necessary
# 
# Hint: use **describe()**

# In[ ]:





# #### Check Missing Values

# In[ ]:





# ### Question 4
# #### Check how many `?` are there in Bare Nuclei feature (they are also unknown or missing values). 

# In[ ]:





# #### Replace them with the 'top' value of the describe function of Bare Nuclei feature
# 
# Hint: give value of parameter include='all' in describe function

# In[ ]:





# ### Question 5
# #### Find the distribution of target variable (Class) 

# In[ ]:





# #### Plot the distribution of target variable using histogram

# In[ ]:





# #### Convert the datatype of Bare Nuclei to `int`

# In[ ]:





# ## Scatter plot

# ### Question 6
# Plot Scatter Matrix to understand the distribution of variables and check if any variables are collinear and drop one of them.

# In[ ]:





# ## Train test split

# ### Question 7
# #### Divide the dataset into feature set and target set

# In[ ]:





# #### Divide the Training and Test sets in 70:30 

# In[ ]:





# ## Scale the data

# ### Question 8
# Standardize the data
# 
# Hint: use **StandardScaler()**

# In[ ]:





# ## Build Model

# ### Question 9
# 
# Build the model and train and test on training and test sets respectively using **scikit-learn**.
# 
# Print the Accuracy of the model with different values of **k = 3, 5, 9**
# 
# Hint: For accuracy you can check **accuracy_score()** in scikit-learn

# In[ ]:





# ## Find optimal value of K

# ### Question 10
# Finding Optimal value of k
# 
# - Run the KNN with no of neighbours to be 1, 3, 5 ... 19
# - Find the **optimal number of neighbours** from the above list

# In[ ]:





# ## Plot accuracy

# ### Question 11
# 
# Plot accuracy score vs k (with k value on X-axis) using matplotlib.

# In[ ]:




