import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression



#a= ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
#C, H,I,j,k,l,m,n,o
a= ['C','H','I','J','K','L','M','O']
n= len(a)
max_accuracy=0
maxb=[]
data= pd.read_csv('/home/hadoop/Desktop/MakemyTrip/datasete9dc3ed/dataset/train.csv',index_col=0)
data.dropna(how="any",inplace=True)
label_encoder = LabelEncoder()
data['A'] = label_encoder.fit_transform(data['A'])
data['D'] = label_encoder.fit_transform(data['D'])
data['E'] = label_encoder.fit_transform(data['E'])
data['F'] = label_encoder.fit_transform(data['F'])
data['G'] = label_encoder.fit_transform(data['G'])
data['I'] = label_encoder.fit_transform(data['I'])
data['J'] = label_encoder.fit_transform(data['J'])
data['L'] = label_encoder.fit_transform(data['L'])
data['M'] = label_encoder.fit_transform(data['M'])

for i in range(1,2**n):
	b=[]
	for j in range(0,n):
		if i&1<<j!=0:
			b.append(a[j])
	X= data[b].as_matrix()
	y= data['P']
#	X =  StandardScaler().fit_transform(X)
	Xtrain, Xtest, ytrain, ytest= train_test_split(X,y,test_size=0.33,random_state=42)
#	clf= DecisionTreeClassifier(random_state=1,max_depth=7,min_samples_split=2)
	clf= SVC(kernel="linear")
#	clf= LogisticRegression()
	clf.fit(Xtrain,ytrain)
	testing_accuracy= clf.score(Xtest,ytest)
	s="hello"
	if testing_accuracy>max_accuracy:
		max_accuracy=testing_accuracy
		maxb=b


print max_accuracy
print maxb