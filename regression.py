import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
weather={'day':[2,4,6,8,10,12,14,16,18,20],'rain':[1,2,3,4,5,6,7,8,9,10]}
df=pd.DataFrame(weather)
#df['sas']=df['day'].shift(-1)
#df.dropna(inplace=True)
#print(df)
lm=LinearRegression()
#ww=np.mean((df.day-df.rain)**2)
#print(ww)
#plt.scatter(df.day,df.rain)
#plt.show()
x=np.array(df['rain']) #converts the entire dataset to array
y=np.array(df['day'])  #x-input....y-output
#print(x[1][1])
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)#random divide x and y into 2 half
y_train.reshape(1,-1)
lm.fit(x_train.reshape(len(x_train),1), y_train) #linearfitfirst part of half
dd=lm.score(x_test.reshape(len(x_test),1),y_test) #test if the earlier linearfitting works to what extent in 2nd half
print(dd)#AnswershouldbeOne CauseFuckYou i have considerd alinear dataset
print(len(x_train),len(y_train))
 