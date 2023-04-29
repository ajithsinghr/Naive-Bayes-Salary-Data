# -*- coding: utf-8 -*-


#importing the data

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("D:\\Assignments\\Naive bayes\\SalaryData_Test.csv")
df.head()
list(df)
df.info()
df.describe()
df.dtypes

#Finding the special characters in the data frame 

df.isin(['?']).sum(axis=0)
print(df[0:5])

df.native.value_counts()
df.native.unique()

df.workclass.value_counts()
df.workclass.unique()

df.occupation.value_counts()
df.occupation.unique()

df.sex.value_counts()


#finding categorical and numerical
cat = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(cat)))
print('The categorical variables are :\n\n', cat)

#find numerical variables
num = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(num)))
print('The numerical variables are :', num)



# check if there are any missing values missing values in categorical variables

df[cat].isnull().sum()

df[num].isnull().sum()


#visualisation
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df)

t1 = pd.crosstab(index=df["education"],columns=df["workclass"])
t1.plot(kind='bar')

t2 = pd.crosstab(index=df["education"],columns=df["Salary"])
t2.plot(kind='bar')


t3 = pd.crosstab(index=df["sex"],columns=df["race"])
t3.plot(kind='bar')


t4 = pd.crosstab(index=df["maritalstatus"],columns=df["sex"])
t4.plot(kind='bar')

df["age"].hist()
df["educationno"].hist()
df["capitalgain"].hist()
df["capitalloss"].hist()
df["hoursperweek"].hist()


# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))
# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()


#splitting the data into x and y

X = df.drop(['Salary'], axis=1)

y = df['Salary']

#======================================================================================================================

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#======================================================================================================================

# display categorical variables
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
categorical

# display numerical variables
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical


#Data Transformation
#pip install category_encoders
import category_encoders as ce


encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_train.head()



cols = X_train.columns
 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])

X_train.head()



#model fitting

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, y_train)

y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)

from sklearn.metrics import accuracy_score
print("trainin score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))


from sklearn.naive_bayes import BernoulliNB
bn = BernoulliNB()

bn.fit(X_train, y_train)

y_pred_train = bn.predict(X_train)
y_pred_test = bn.predict(X_test)

from sklearn.metrics import accuracy_score
print("trainin score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))



from sklearn.svm import SVC

svc_class = SVC(kernel='linear')

svc_class.fit(X_train,y_train)

y_predict_train = svc_class.predict(X_train)
y_predict_test = svc_class.predict(X_test)


print("training score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

#prediction
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)

print("trainin score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))

#svc and logreg is best prediction models among the builted models