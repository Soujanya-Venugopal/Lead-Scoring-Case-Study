#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Case Study - Problem Statement

# Problem Statement Summary: Lead Conversion Rate Improvement for X Education
# 
# X Education, an online course provider, faces a challenge in optimizing its lead conversion process. Despite generating numerous leads daily, the conversion rate stands at a suboptimal 30%. The company aims to enhance this by identifying and prioritizing potential leads, referred to as 'Hot Leads,' to increase the lead conversion rate to approximately 80%.
# 
# Goals:
# 
# Develop a logistic regression model to assign lead scores ranging from 0 to 100. A higher score indicates a higher likelihood of conversion, enabling the sales team to focus efforts on potential leads.

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('leads.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.describe()


# In[7]:


data.info()


# In our dataset, we have information that falls into different categories (like types of leads or sources). To use this data in our model, we'll create separate categories for each of these types, and we call them dummy variables.
# 
# Additionally, there are some missing values in our data. We'll need to decide what to do with these gaps. It's a bit like completing a puzzle - we have to figure out the missing pieces before we can use the information effectively.

# # Data Cleaning and Preparation

# In[9]:


data.isnull().sum().sort_values(ascending=False)


# Looking at our data, we notice that some columns have a lot of missing information. Imagine these columns are like pages in a book where many words are blank. Since we have a total of 9000 pages (data points), we decide to throw away the pages (columns) that have more than 3000 blank words because they won't help us much. It's like getting rid of the parts of the book that don't give us useful information.

# In[10]:


# Drop all the columns in which greater than 3000 missing values are present

for col in data.columns:
    if data[col].isnull().sum() > 3000:
        data.drop(col, 1, inplace=True)


# In[11]:


# Check the number of null values again

data.isnull().sum().sort_values(ascending=False)


# In[12]:


#checking value counts of "City" column
data['City'].value_counts(dropna=False)


# Mumbai leads the chart with the highest number of leads.
# 
# As you can infer, the variable 'City' doesn't contribute meaningfully to our analysis. Hence, it is advisable to remove it from consideration.

# In[13]:


# dropping the "City" feature
data.drop(['City'], axis = 1, inplace = True)


# In[14]:


#checking value counts of "Country" column
data['Country'].value_counts(dropna=False)


# In[15]:


# dropping the "Country" feature
data.drop(['Country'], axis = 1, inplace = True)


# In[16]:


# Let's now check the percentage of missing values in each column

round(100*(data.isnull().sum()/len(data.index)), 2)


# In[17]:


# Checking the number of null values again
data.isnull().sum().sort_values(ascending=False)


# # Visualizing the features with Select values
# 

# In[19]:


# Function to create countplot
def countplot(x, fig):
    plt.subplot(2, 2, fig)
    sns.countplot(data=data[data[x] != 'Select'], x=x)
    plt.title('Count across' + ' ' + x, size=16)
    plt.xlabel(x, size=14)
    plt.xticks(rotation=90)

# Create a figure
plt.figure(figsize=(15, 10))

# Call the function for different columns
countplot('How did you hear about X Education', 1)
countplot('Lead Profile', 2)
countplot('Specialization', 3)

# Show the plots
plt.show()


# Some columns in our data have a value called 'Select,' indicating that the student didn't choose any option for that particular category. This 'Select' is similar to having no information, like a blank answer. To understand how common this is, we want to count how many times 'Select' appears in each of these columns. This helps us identify how often students haven't chosen an option in those specific categories.

# In[20]:


# Get the value counts of all the columns

for column in data:
    print(data[column].astype('category').value_counts())
    print('___________________________________________________')


# The following three columns now have the level 'Select'. Let's check them once again.

# In[21]:


data['Lead Profile'].astype('category').value_counts()


# In[22]:


data['How did you hear about X Education'].value_counts()


# In[23]:


data['Specialization'].value_counts()


# In our data, the columns 'Lead Profile' and 'How did you hear about X Education' have many rows with the value 'Select,' which doesn't give us useful information. It's like having a choice that doesn't help us understand the data. So, we decide to remove these rows because they won't contribute to our analysis.

# # Visualizing the Features

# In[30]:


# Function to create countplot
def countplot(x, fig):
    plt.subplot(4, 2, fig)
    sns.countplot(data=data, x=x)
    plt.title('Count across' + ' ' + x, size=16)
    plt.xlabel(x, size=14)
    plt.xticks(rotation=90)

# Create a figure
plt.figure(figsize=(18, 25))

# Call the function for different columns
countplot('What matters most to you in choosing a course', 1)
countplot('What is your current occupation', 3)
countplot('Specialization', 5)

# Show the plots
plt.show()


# In[31]:


data.drop(['Lead Profile', 'How did you hear about X Education'], axis = 1, inplace = True)


# When we checked the data, some columns like 'Do Not Call' and others mostly had just one value, which is 'No,' for all data points. Having the same answer for everything doesn't provide useful insights. Therefore, we decide to remove these columns, like 'Do Not Call,' 'Search,' and others, as they won't contribute meaningfully to our analysis. It's like removing unnecessary details that don't add value to our understanding.

# In[32]:


data.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 'Update me on Supply Chain Content', 'Get updates on DM Content', 'I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[33]:


data['What matters most to you in choosing a course'].value_counts()


# In the data, the column 'What matters most to you in choosing a course' is mostly filled with one option, 'Better Career Prospects,' appearing 6528 times. The other two options show up only once, twice, and once, respectively. Since most values are the same, we decide to remove this column too, as it won't provide diverse information for our analysis. It's like removing a column with repetitive details that won't enhance our understanding.

# In[35]:


# Drop the null value rows present in the variable 'What matters most to you in choosing a course'

data.drop(['What matters most to you in choosing a course'], axis = 1, inplace=True)


# In[36]:


data.head()


# # Prepare the data for Modelling

# In[37]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.pairplot(data,diag_kind='kde',hue='Converted')
plt.show()


# In[40]:


xedu = data[['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']]
sns.pairplot(xedu,diag_kind='kde',hue='Converted')
plt.show()


# In[41]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
transformedxedu = pd.DataFrame(pt.fit_transform(xedu))
transformedxedu.columns = xedu.columns
transformedxedu.head()


# In[42]:


sns.pairplot(transformedxedu,diag_kind='kde',hue='Converted')
plt.show()


# In[44]:


# Dropping the null values rows in the column 'What is your current occupation'

data = data[~pd.isnull(data['What is your current occupation'])]


# # # Correlation
# Now, examine the correlations among variables. Due to the high number of variables, it's more practical to review the correlation table rather than creating a heatmap for visualization.

# In[45]:


# Looking at the correlation table
plt.figure(figsize = (10,8))
sns.heatmap(data.corr())
plt.show()


# # Analysing Categorical features

# In[47]:


conv = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']

for i in conv:
    plt.figure(figsize=(15, 5))
    sns.countplot(x=data[i], hue=data['Converted'])
    plt.xticks(rotation=90)
    plt.title('Target variable in ' + i)
    plt.show()


# In[48]:


# Checking the number of null values again
data.isnull().sum().sort_values(ascending=False)


# In[49]:


# Dropping the null values rows in the column 'TotalVisits'

data = data[~pd.isnull(data['TotalVisits'])]


# In[50]:


# Checking the number of null values again
data.isnull().sum().sort_values(ascending=False)


# In[51]:


# Dropping the null values rows in the column 'Lead Source'

data = data[~pd.isnull(data['Lead Source'])]


# In[52]:


# Checking the number of null values again
data.isnull().sum().sort_values(ascending=False)


# In[53]:


# Drop the null values rows in the column 'Specialization'

data = data[~pd.isnull(data['Specialization'])]


# In[54]:


# Checking the number of null values again
data.isnull().sum().sort_values(ascending=False)


# In[55]:


print(len(data.index))
print(len(data.index)/9240)


# We still have around 69% of the rows which seems good enough.

# In[56]:


data.head()


# The variables Prospect ID and Lead Number won't help in our analysis, so it's best to remove these two from our data.

# In[57]:


data.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[58]:


data.head()


# # Creation of Dummy Variables
# 
# Now, let's handle categorical variables in the dataset. Begin by identifying which variables are categorized. This step is crucial for creating dummy variables to represent different categories.

# In[59]:


# Check the columns which are of type 'object'

temp = data.loc[:, data.dtypes == 'object']
temp.columns


# In[60]:


# Create dummy variables using the 'get_dummies' command
dummy = pd.get_dummies(data[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity', 'What is your current occupation','A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first=True)

# Add the results to the master dataframe
data = pd.concat([data, dummy], axis=1)


# In[61]:


# Creating dummy variable separately for the variable 'Specialization' since it has the level 'Select' which is useless so we
# drop that level by specifying it explicitly

dummy_spl = pd.get_dummies(data['Specialization'], prefix = 'Specialization')
dummy_spl = dummy_spl.drop(['Specialization_Select'], 1)
data = pd.concat([data, dummy_spl], axis = 1)


# In[62]:


#Drop the variables for which the dummy variables have been created

data = data.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity','Specialization', 'What is your current occupation','A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[63]:


data.head()


# # Test-Train Split
# The next step is to split the dataset into training an testing sets.

# In[66]:


# Import the required library

from sklearn.model_selection import train_test_split


# In[67]:


X = data.drop(['Converted'], 1)
X.head()


# In[68]:


y = data['Converted']

y.head()


# In[69]:


# Split the dataset into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# # Scaling
# In our dataset, some numeric variables have different scales. To address this, we'll scale these variables. This ensures they're on a consistent scale, making them comparable and aiding analysis.

# In[70]:


# Import MinMax scaler

from sklearn.preprocessing import MinMaxScaler


# In[71]:


scaler = MinMaxScaler()

X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# # Correlation
# Now, examine the correlations among variables. Due to the high number of variables, it's more practical to review the correlation table rather than creating a heatmap for visualization.

# In[72]:


# Looking at the correlation table
plt.figure(figsize = (20,10))
sns.heatmap(data.corr())
plt.show()


# # Model Building
# Now, let's transition to building the model. Given the abundance of variables in the dataset, we need a focused set for analysis. The approach here is to choose a subset of features using Recursive Feature Elimination (RFE). This method helps us narrow down to a smaller, more manageable set of impactful variables.

# In[73]:


# Import 'LogisticRegression' and create a LogisticRegression object

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[74]:


# Import RFE and select 15 variables
from sklearn.feature_selection import RFE

# Assuming logreg is your logistic regression model
rfe = RFE(estimator=logreg, n_features_to_select=15)  # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[75]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[76]:


col = X_train.columns[rfe.support_]


# Now that RFE has chosen the relevant variables, let's focus on the statistical details like p-values and VIFs. We'll utilize these selected variables to construct a logistic regression model using the statsmodels library. This allows us to delve deeper into the statistical significance and multicollinearity of the chosen features.

# In[77]:


X_train = X_train[col]


# In[78]:


import statsmodels.api as sm


# In[79]:


X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# Several variables have p-values exceeding 0.05, indicating potential insignificance. Before addressing them, let's examine the Variance Inflation Factors (VIFs). This will help us understand if there's high correlation among variables, impacting their reliability in the model.

# In[80]:


# Import 'variance_inflation_factor'

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[81]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Most VIFs appear reasonable, but three variables stand out. To address this, let's start by removing the 'Lead Source_Reference' variable. This decision is based on its elevated p-value and VIF, suggesting potential issues with significance and multicollinearity, which could affect model reliability.

# In[82]:


X_train.drop('Lead Source_Reference', axis = 1, inplace = True)


# In[83]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# Remove 'Lead Profile_Dual Specialization Student' from the variables.

# # Checking VIF

# In[84]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Drop variables starting with 'Last Notable Activity_Had a Phone Conversation'.

# In[85]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# In[86]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# Drop 'What is your current occupation_Housewife'.

# In[87]:


X_train.drop('What is your current occupation_Housewife', axis = 1, inplace = True)


# In[88]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# Drop What is your current occupation_Working Professional.

# In[89]:


X_train.drop('What is your current occupation_Working Professional', axis = 1, inplace = True)


# In[90]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[91]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Model Evaluation
# With satisfactory p-values and VIFs for all variables, we proceed to predictions using this refined set of features. The model evaluation indicates that the chosen features contribute meaningfully. Now, we can apply these features to make predictions and assess the model's performance.

# In[92]:


# Use 'predict' to predict the probabilities on the train set

y_train_pred = res.predict(sm.add_constant(X_train))
y_train_pred[:10]


# In[93]:


# Reshaping it into an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# Make a table with real conversions and predicted probabilities.

# In[94]:


# Create a new dataframe containing the actual conversion flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# Add 'Predicted' column: 1 if Paid_Prob > 0.5, else 0.

# In[95]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# Now, assess the model after using probabilities for conversion predictions.

# In[96]:


# Import metrics from sklearn for evaluation

from sklearn import metrics


# In[97]:


# Create confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[98]:


# Let's check the overall accuracy

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[99]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[100]:


# Calculate the sensitivity

TP/(TP+FN)


# In[101]:


# Calculate the specificity

TN/(TN+FP)


# The chosen threshold of 0.5 was somewhat arbitrary for a preliminary model check. To enhance results, we aim to optimize the threshold. Initially, we'll plot a Receiver Operating Characteristic (ROC) curve to assess the Area Under the Curve (AUC). This helps us determine an optimal threshold for improved model performance.

# In[102]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[103]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[104]:


# Import matplotlib to plot the ROC curve

import matplotlib.pyplot as plt


# In[105]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# The ROC curve's Area Under the Curve (AUC) is 0.86, indicating a robust model. Now, let's examine the balance between sensitivity and specificity to pinpoint the best cutoff point. This tradeoff assessment will help us determine the optimal threshold, ensuring an effective balance between correctly identifying positives and negatives.

# In[106]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[107]:


# Assuming y_train_pred_final is your DataFrame with predicted probabilities and actual values

cutoff_df = pd.DataFrame(columns=['prob', 'accuracy', 'sensi', 'speci'])

num = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for i in num:
    cm1 = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i].apply(lambda x: 1 if x > i else 0))
    total1 = sum(sum(cm1))
    accuracy = (cm1[0, 0] + cm1[1, 1]) / total1
    speci = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    sensi = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    cutoff_df.loc[i] = [i, accuracy, sensi, speci]

print(cutoff_df)


# In[108]:


cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[109]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.42 else 0)

y_train_pred_final.head()


# In[110]:


metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[111]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[112]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[113]:


#Sensitivity
TP/(TP+FN)


# In[114]:


#Specificity
TN/(TN+FP)


# # Predictions on the Test Set¶

# In[115]:


# Scale the test set as well using just 'transform'

X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[116]:


# Select the columns in X_train for X_test as well

X_test = X_test[col]
X_test.head()


# In[117]:


# Add a constant to X_test

X_test_sm = sm.add_constant(X_test[col])


# In[118]:


# Check X_test_sm

X_test_sm


# In[119]:


# Drop the required columns from X_test as well

X_test.drop(['Lead Source_Reference', 'What is your current occupation_Housewife', 
             'What is your current occupation_Working Professional', 'Last Notable Activity_Had a Phone Conversation'], 1, inplace = True)


# In[120]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(sm.add_constant(X_test))


# In[121]:


y_test_pred[:10]


# In[122]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[123]:


# Let's see the head

y_pred_1.head()


# In[124]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[125]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[126]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[127]:


# Check 'y_pred_final'

y_pred_final.head()


# In[128]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[129]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[130]:


# Make predictions on the test set using 0.42 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[131]:


# Check y_pred_final

y_pred_final.head()


# In[132]:


metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[133]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[134]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[135]:


# Calculate sensitivity
TP / float(TP+FN)


# In[136]:


# Calculate specificity
TN / float(TN+FP)


# # Precision-Recall View
# 

# In[137]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[138]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[139]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# # Precision and recall tradeoff 

# In[140]:


from sklearn.metrics import precision_recall_curve


# In[141]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[142]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[143]:


plt.plot(thresholds, p[:-1], "b-")
plt.plot(thresholds, r[:-1], "y-")
plt.show()


# In[144]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.44 else 0)

y_train_pred_final.head()


# In[145]:


# check the accuracy

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[146]:


# Create the confusion matrix

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[147]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[148]:


# Calculate Precision

TP/(TP+FP)


# In[149]:


# Calculate Recall

TP/(TP+FN)


# Making Predictions on the Test Set
# Let's now make predicitons on the test set.

# In[150]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(sm.add_constant(X_test))


# In[151]:


y_test_pred[:10]


# In[152]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[153]:


y_pred_1.head()


# In[154]:


y_test_df = pd.DataFrame(y_test)


# In[155]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[156]:


y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[157]:


y_pred_final.head()


# In[158]:


y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'}) #Renaming Column


# In[159]:


y_pred_final.head()


# In[160]:


print(y_pred_final.columns)


# In[161]:


threshold = 0.5
y_pred_binary = (y_pred_final.Conversion_Prob >= threshold).astype(int)
accuracy = metrics.accuracy_score(y_pred_final['Converted'], y_pred_binary)
print('Accuracy:', accuracy)


# In[162]:


threshold = 0.5
# Convert probabilities to binary predictions based on the threshold
y_pred_binary = (y_pred_final.Conversion_Prob >= threshold).astype(int)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_pred_final['Converted'], y_pred_binary)

print('Confusion Matrix:')
print(confusion_mat)


# In[163]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[164]:


TP/(TP+FP)


# In[165]:


TP/(TP+FN)


# # Summary
# In the early stages of lead generation, numerous potential customers are identified, but only a small portion eventually become paying clients. To improve lead conversion, it's crucial to nurture potential leads effectively during the middle stage. This involves educating them about the product, maintaining consistent communication, and sorting out the most promising prospects. Factors like 'TotalVisits,' 'Total Time Spent on Website,' and 'Page Views Per Visit' play a significant role in predicting a lead's conversion probability.
# 
# Once you've identified the best prospects, it's essential to maintain a list of these leads. This list serves as a valuable resource for informing them about new courses, services, job offers, and future educational opportunities. Monitoring each lead closely allows you to tailor the information you send based on their interests. Developing a thoughtful plan to address the unique needs of each lead contributes to successful lead capture.
# 
# Focusing on converted leads is key. Engage in question-and-answer sessions with leads to gather essential information. Through further inquiries and appointments, determine their intentions and readiness to join online courses. This proactive approach enhances the understanding of each lead, facilitating more effective communication and increasing the likelihood of successful conversions.

# In[ ]:




