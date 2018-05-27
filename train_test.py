import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


train_data= pd.read_csv('/home/hadoop/Desktop/MakemyTrip/datasete9dc3ed/dataset/train.csv',index_col=0)
test_data= pd.read_csv('/home/hadoop/Desktop/MakemyTrip/datasete9dc3ed/dataset/test.csv',index_col=0)

#print test_data.count()
#print test_data.shape

train_data.dropna(how="any",inplace=True)
#test_data.dropna(how="any").shape
#test_data.A.value_counts(dropna=False)
#null_columns=test_data.columns[test_data.isnull().any()]

null_test_data= test_data[test_data.isnull().any(axis=1)]
test_data.dropna(how="any",inplace=True)

label_encoder = LabelEncoder()
train_data['A'] = label_encoder.fit_transform(train_data['A'])
train_data['D'] = label_encoder.fit_transform(train_data['D'])
train_data['E'] = label_encoder.fit_transform(train_data['E'])
train_data['F'] = label_encoder.fit_transform(train_data['F'])
train_data['G'] = label_encoder.fit_transform(train_data['G'])
train_data['I'] = label_encoder.fit_transform(train_data['I'])
train_data['J'] = label_encoder.fit_transform(train_data['J'])
train_data['L'] = label_encoder.fit_transform(train_data['L'])
train_data['M'] = label_encoder.fit_transform(train_data['M'])

test_data['A'] = label_encoder.fit_transform(test_data['A'])
test_data['D'] = label_encoder.fit_transform(test_data['D'])
test_data['E'] = label_encoder.fit_transform(test_data['E'])
test_data['F'] = label_encoder.fit_transform(test_data['F'])
test_data['G'] = label_encoder.fit_transform(test_data['G'])
test_data['I'] = label_encoder.fit_transform(test_data['I'])
test_data['J'] = label_encoder.fit_transform(test_data['J'])
test_data['L'] = label_encoder.fit_transform(test_data['L'])
test_data['M'] = label_encoder.fit_transform(test_data['M'])

clf= clf= LogisticRegression()
clf.fit(train_data[['D','E','F','I','K','M','O']],train_data['P'])

ypredict= clf.predict(test_data[['D','E','F','I','K','M','O']])
ouput={}
for i in range(len(test_data.index)):
    ouput[test_data.index[i]]= ypredict[i]

clf2=clf= LogisticRegression()
clf2.fit(train_data[['C','H','I','K','O']],train_data['P'])
null_test_data['I'] = label_encoder.fit_transform(null_test_data['I'])
null_test_data['J'] = label_encoder.fit_transform(null_test_data['J'])
ypredict2= clf2.predict(null_test_data[['C','H','I','K','O']])

for i in range(len(null_test_data.index)):
    ouput[null_test_data.index[i]]= ypredict2[i]


df= pd.DataFrame(ouput.items(), columns=['id', 'P'])
df.to_csv("out.csv",index=False)

