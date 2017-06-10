import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import datetime
from datetime import datetime
weather={'day':[1,2,3,4,5,6,7,8,9,10],'rain':[11,23,30,41,55,77,74,80,96,108]}
gurren={'future':[11,12,13,14]}
j=pd.DataFrame(gurren)
df=pd.DataFrame(weather)
ee=np.array(j['future'])
#df['sas']=df['day'].shift(-1)
#df.dropna(inplace=True)
#print(df)
lm=LinearRegression()
#ww=np.mean((df.day-df.rain)**2)
#print(ww)
#plt.scatter(df.day,df.rain)
#plt.show()
x=np.array(df['day']) #converts the entire dataset to array
y=np.array(df['rain'])      #x-input....y-output
#print(x[1][1])
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)#random divide x and y into 2 half
y_train.reshape(1,-1)
lm.fit(x_train.reshape(len(x_train),1), y_train) #linearfitfirst part of half
accuracy=lm.score(x_test.reshape(len(x_test),1),y_test) #test if the earlier linearfitting works to what extent in 2nd half
forecast=lm.predict(ee.reshape(len(ee),1))
forecast=np.array(forecast)
df["forecast"]=np.nan
last_date=df.iloc[-1].name
last_unix=last_date 
one_day=1
next_unix=last_unix+one_day
for i in forecast:
	next_date=next_unix
	df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
	next_unix+=one_day
print(df)
df["rain"].plot()
df["forecast"].plot()
plt.show() 