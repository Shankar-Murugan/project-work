#!/usr/bin/env python
# coding: utf-8

# **KNN**
# 
# According to this algorithm, which is used in classification, feature extraction during classification is used to look at the closeness of the new individual to be categorized to k of the previous individuals.
# For example, you want to classify a new element for k = 3. in this case the nearest 3 of the old classified elements are taken. If these elements are included in the class, the new element is also included in that class. The euclide distance can be used in the distance calculation.

# # Project - Classify Type of Patient from the biomechanical features of orthopedic patients.

# # Install the necessary libraries
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn import metrics


# # import data 3Classdata.csv for 3 Class Classifcation.
# 

# In[2]:


Class_1 = pd.read_csv("3Classdata.csv")


# # Explore the data set.Get necessary information about the data.

# Look at the head and tail of dataset.
# Find the missing value.
# Look at the unique values of class values.
# Look at the distribution of class values and other attributes.
# Get the datatype information about the dataset
# Plot the distribution of different classes for pelvic_radius and sacral_slope for visualization.

# In[3]:


Class_1.head()


# In[4]:


Class_1.tail()


# In[5]:


Class_1.isnull().sum()


# In[6]:


Class_1.shape


# In[7]:


Class_1.columns


# In[8]:


Class_1.info()


# In[9]:


Class_1 = Class_1.rename(columns = {"class": "Class"})
Class_1.info()


# In[10]:


Class_1['Class'] = Class_1.Class.astype('category')
Class_1.dtypes


# In[11]:


Class_1.describe()


# In[12]:


Class_1.describe().T


# In[14]:


Class_1.groupby(["Class"]).count()


# In[15]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(Class_1, hue = 'Class')


# # Encode the Class variable to integer.

# In[16]:


from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
Class_1["Class"] = lb_make.fit_transform(Class_1["Class"])
Class_1.head()


# # Create the X(Feature-set) and Y(Target-set) sets for your Data.

# In[18]:


X = Class_1.drop(labels= "Class" , axis = 1)
X.head()


# In[19]:


y = Class_1[["Class"]]
y.head()


# # Normalize your Data (X) to get values between 0 to 1.

# In[20]:


X = X.apply(zscore)
X.head()


# # Split the dat as train and test with a ratio of 70:30.

# In[22]:


test_size = 0.30 
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# # Build the KNN model using Train Dataset and predict the class on test dataset.

# In[23]:


knn = KNeighborsClassifier()
knn


# In[24]:


knn.fit(X_train, y_train)


# In[25]:


X_test.shape


# In[26]:


X_train.shape


# In[27]:


knn.predict(X_test)


# In[28]:


y_test.head(10)


# In[29]:


y_test.shape


# In[31]:


predicted_labels = knn.predict(X_test)


# # Calculate the performance score of of your classification on test dataset.
# Hint- You can use knn.score( ) function.

# In[32]:


knn.score(X_test, y_test)


# In[33]:


metrics.confusion_matrix(y_train, knn.predict(X_train))


# # What is the best K value for your classifcation?
# #Find at which k value you get the best score.

# In[34]:


X_train.shape[0]


# In[35]:


maxK = int(np.sqrt(X_train.shape[0]))
print(maxK)


# In[36]:


optimalK = 1
optimalTrainAccuracy = 0


# In[37]:


for k_i in range(maxK):
    if(((k_i % 2) != 0) & (k_i > 1)):
        knn = KNeighborsClassifier(n_neighbors=k_i)
        knn.fit(X_train, y_train)
        if(knn.score(X_train, y_train) > optimalTrainAccuracy):
            optimalK = k_i
            optimalTrainAccuracy = knn.score(X_train, y_train)
print((optimalK, optimalTrainAccuracy))


# In[38]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# In[39]:


metrics.confusion_matrix(y_test, knn.predict(X_test))


# # import data 2Classdata.csv for 2 Class Classifcation and repeat all the steps which are given above.

# In[41]:


Class_2 = pd.read_csv("2Classdata.csv")


# # Explore the data set.Get necessary information about the data.

# Look at the head and tail of dataset.
# Find the missing value.
# Look at the unique values of class values.
# Look at the distribution of class values and other attributes.
# Get the datatype information about the dataset
# Plot the distribution of different classes for pelvic_radius and sacral_slope for visualization.

# In[42]:


Class_2.head()


# In[43]:


Class_2.tail()


# In[44]:


Class_2.isnull().sum()


# In[45]:


Class_2.shape


# In[46]:


Class_2.columns


# In[47]:


Class_2.describe()


# In[48]:


Class_2.describe().T


# In[66]:


Class_2.head()


# In[67]:


Class_2.groupby(["Class"]).count()


# In[62]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(Class_2, hue = 'Class')


# ## Encode the Class variable to integer.

# In[68]:


Class_2 = Class_2.rename(columns = {"class": "Class"})
Class_2.info()


# In[69]:


Class_2['Class'] = Class_2.Class.astype('category')
Class_2.dtypes


# In[70]:


X = Class_2.drop(labels= "Class" , axis = 1)
X.head()


# In[71]:


y = Class_2[["Class"]]
y.head()


# In[72]:


y = pd.get_dummies(y, drop_first=True)
y.head()


# ## Create the X(Feature-set) and Y(Target-set) sets for your Data.

# In[73]:


X.head()


# In[58]:


y.head()


# ## Normalize your Data (X) to get values between 0 to 1.

# In[74]:


X = X.apply(zscore)
X.head()


# ## Split the dat as train and test with a ratio of 70:30.

# In[75]:


test_size = 0.30 
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# ## Build the KNN model using Train Dataset and predict the class on test dataset.

# In[76]:


khn = KNeighborsClassifier()
khn


# In[77]:


khn.fit(X_train, y_train)


# In[78]:


X_test.shape


# In[79]:


X_train.shape


# In[80]:


khn.predict(X_test)


# In[81]:


y_test.head(10)


# In[82]:


y_test.shape


# In[83]:


predicted_labels = khn.predict(X_test)


# # Calculate the performance score of of your classification on test dataset.
# 

# Hint- You can use knn.score( ) function.

# In[84]:


khn.score(X_test, y_test)


# In[85]:


metrics.confusion_matrix(y_train, khn.predict(X_train))


# ## What is the best K value for your classifcation?
# 

# #Find at which k value you get the best score.

# In[86]:


X_train.shape[0]


# In[87]:


maxK1 = int(np.sqrt(X_train.shape[0]))
print(maxK1)


# In[88]:


optimal_K = 1
optimal_TrainAccuracy = 0


# In[89]:


for k_i in range(maxK1):
    if(((k_i % 2) != 0) & (k_i > 1)):
        khn = KNeighborsClassifier(n_neighbors=k_i)
        khn.fit(X_train, y_train)
        if(khn.score(X_train, y_train) > optimal_TrainAccuracy):
            optimal_K = k_i
            optimal_TrainAccuracy = khn.score(X_train, y_train)
print((optimalK, optimalTrainAccuracy))


# In[90]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# In[91]:


metrics.confusion_matrix(y_test, khn.predict(X_test))

