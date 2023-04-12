#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[9]:


kt = pd.read_csv(w//'file2.csv')


# In[10]:


kt.head()


# In[12]:


df.shape


# In[13]:


df.info()


# In[14]:


import datetime


# In[15]:


df['date_time_of_pickup']=pd.to_datetime(df['date_time_of_pickup'])-datetime.timedelta(hours=4)


# In[16]:


df.info()


# In[17]:


df.head()


# In[18]:


df['date_time_of_pickup'].dt.month


# In[19]:


df['year']=df['date_time_of_pickup'].dt.year
df['Month']=df['date_time_of_pickup'].dt.month
df['Day']=df['date_time_of_pickup'].dt.day
df['Hours']=df['date_time_of_pickup'].dt.hour
df['Minutes']=df['date_time_of_pickup'].dt.minute


# In[20]:


df.head()


# In[21]:


import numpy as np


# In[22]:


df['mornight']=np.where(df['Hours']<12,0,1)


# In[23]:


df.head()


# In[24]:


df.drop('date_time_of_pickup',axis=1,inplace=True)


# In[25]:


df.head()


# In[26]:


df['amount'].unique()


# In[27]:





# In[ ]:





# In[ ]:





# In[29]:


from sklearn.metrics.pairwise import haversine_distances
from math import radians
newdelhi=[28.6139,77.2090]
banglore=[12.9716,77.5946]


# In[30]:


newdelhi_in_radian=[radians(_) for _ in newdelhi]
banglore_in_radian=[radians(_) for _ in banglore]


# In[31]:


result=haversine_distances([newdelhi_in_radian,banglore_in_radian])

result*6371
# In[33]:


np.radians(df['latitude_of_dropoff'])-df['latitude_of_pickup']


# In[37]:


def haversine(df):
    lat1 =np.radians(['latitude_of_pickup'])
    lat2 =np.radians(['latitude_of_dropoff'])
    dlat =np.radians(df['latitude_of_dropoff']-df["latitude_of_pickup"])
    dlong =np.radians(df["longitude_of_dropoff"]-df["longitude_of_pickup"])
    a = np.sin(dlat/2)**2+np.cos(lat1) * np.cos(lat2) * np.sin(dlong/2)**2
    c =2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r =6371
    return c * r
    


# In[40]:


df['Total_distance']=haversine(df)


# In[30]:


df.head()


# In[32]:


def haversine(df):
    lat1 =np.radians(['latitude_of_pickup'])
    lat2 =np.radians(['latitude_of_dropoff'])
    dlat =np.radians(df['latitude_of_dropoff']-df["latitude_of_pickup"])
    dlong =np.radians(df["longitude_of_dropoff"]-df["longitude_of_pickup"]


# In[41]:


from sklearn.metrics.pairwise import haversine_distances
from math import radians
newdelhi = [28.6139, 77.2090],
bangalore = [12.9716, 77.5946]
    


# In[ ]:


df.drop(["longitude_of_pickup","latitude_of_pickup","longitude_of_dropoff","latitude_of_dropoff"],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


import json
records=json.loads(df.T.to_json()).values()


# In[ ]:


get_ipython().system('pip install pymongo')


# In[ ]:


import pymongo
client = pymongo.MongoClient('mongodb:localhost:27017')
db=client["newyorktaxi"]
col=db["rides"]


# In[ ]:


rcords


# In[ ]:


col.insert_many(records)


# In[ ]:


df.head()


# In[ ]:


df.to_csv('final_data.csv')


# In[ ]:


x=df.iloc(:,1,:)
y=df.iloc(:,0)


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model =  ExtraTreesRegressormodel.fit(x,y)




# In[ ]:


feat_importances=pd.series(model.feature_importances_,index=x.colums)
feat_importances.nlargest(.plot(kind='barh'))
plt.show


# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y.train,y.test=train_test_split(x,y,test_size=0,3,random_state=100)


# In[ ]:


import xgboost


# In[ ]:


regressor=xgboost.xGBRegressor()
regressor.fit(x_train,y_train)


# In[ ]:


y_pred=regressor.predict(x_test)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.distplot(y_test-y_pred)


# In[ ]:


plt.scatter(y_test,y_pred)


# In[ ]:


from sklearn import metrics
    print('R square:', np.sqrt(metrics.r2_score(y_test, y_pred)))
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
    print(n_estimators)


# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
    # Various learning rate parameters
    learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
    # max_depth.append(None)
    #Subssample parameter values
    subsample=[0.7,0.6,0.8]
    # Minimum child weight parameters\
    min_child_weight=[3,4,5,6,7]


# In[ ]:


random_grid = {'n_estimators': n_estimators
        'learning_rate': learning_rate
        'max_depth': max_depth
        'subsample': subsample
        'min_child_weight': min_child_weight}
    
print(random_grid)


# In[ ]:


regressor=xgboost.XGBRegressor()


# In[ ]:


xg_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter =5, cv = 5, verbose=2, random_state=42, n_jobs = -1)


# In[ ]:


xg_random.fit(x_train,y_train)
Fitting 5 folds for each of 5 candidates, totalling 25 fits
      [CV] subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05


# In[ ]:


[CV]  subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05, total= 2.2min
      [CV] subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05 


# In[ ]:


[CV]  subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05, total= 2.2min
      [CV] subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05
      [CV]  subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05, total= 2.1min
      [CV] subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05 
      [CV]  subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05, total= 2.2min
      [CV] subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05 
      [CV]  subsample=0.8, n_estimators=1100, min_child_weight=6, max_depth=25, learning_rate=0.05, total= 2.2min
      [CV] subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5, total= 1.4min
      [CV] subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5, total= 1.1min
      [CV] subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5, total= 1.4min
      [CV] subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5, total= 1.1min
      [CV] subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.8, n_estimators=900, min_child_weight=7, max_depth=30, learning_rate=0.5, total= 1.4min
      [CV] subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5, total=  29.1s
      [CV] subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5, total=  28.3s
      [CV] subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5, total=  30.2s
      [CV] subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5, total=  25.7s
      [CV] subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5 
      [CV]  subsample=0.7, n_estimators=300, min_child_weight=3, max_depth=30, learning_rate=0.5, total=  32.4s
      [CV] subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5 
      [CV]  subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5, total=  32.1s
      [CV] subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5 
      [CV]  subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5, total=  32.5s
      [CV] subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5 
      [CV]  subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5, total=  32.1s
      [CV] subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5 
      [CV]  subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5, total=  32.4s
      [CV] subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5 
      [CV]  subsample=0.6, n_estimators=300, min_child_weight=7, max_depth=25, learning_rate=0.5, total=  32.4s
      [CV] subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3 
      [CV]  subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3, total= 1.1min
      [CV] subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3 
      [CV]  subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3, total= 1.2min
      [CV] subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3 
      [CV]  subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3, total= 1.1min
      [CV] subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3 
      [CV]  subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3, total= 1.2min
      [CV] subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3 
      [CV]  subsample=0.6, n_estimators=1000, min_child_weight=7, max_depth=15, learning_rate=0.3, total= 1.1min


# In[ ]:


xg_random.best_params_


# In[ ]:


y_pred=xg_random.predict(X_test)


# In[ ]:


sns.distplot(y_test-y_pred)


# In[ ]:


plt.scatter(y_test,y_pred)


# In[ ]:


from sklearn import metrics
    print('R square:', np.sqrt(metrics.r2_score(y_test, y_pred)))
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


X_train.shape[1]


# In[ ]:


y_train.shape[1]


# In[ ]:


import tensorflow.keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LeakyReLU,PReLU,ELU
    from tensorflow.keras.layers import Dropout


# In[ ]:


NN_model = Sequential()


# In[ ]:


NN_model = Sequential()
    
    # The Input Layer 
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
    # The Hidden Layers 
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    
    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    
    # Compile the network 
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()
    
    # Fitting the ANN to the Training set
    model_history=NN_model.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 20)


# In[ ]:


prediction=NN_model.predict(X_test)


# In[ ]:


import seaborn as sns
    sns.distplot(y_test.values.reshape(-1,1)-prediction)


# In[ ]:


import matplotlib.pyplot as plt
    plt.scatter(y_test,prediction)


# In[ ]:


from sklearn import metrics
    print('MAE:', metrics.mean_absolute_error(y_test, prediction))
    print('MSE:', metrics.mean_squared_error(y_test, prediction))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[ ]:


from tpot import TPOTRegressor


# In[ ]:


regressor=TPOTRegressor()
regressor.fit(X_train,y_train)


# In[ ]:


tpot = TPOTRegressor(generations=1, population_size=10, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_taxiFaredocumentation_pipeline.py')


# In[ ]:


import numpy as np
    import pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline, make_union
    from tpot.builtins import StackingEstimator
    from tpot.export_utils import set_param_recursive
    
    # NOTE: Make sure that the outcome column is labeled 'target' in the data file
    tpot_data = pd.read_csv('final_data.csv', sep=',', dtype=np.float64)
    features = tpot_data.drop('fare_amount', axis=1)
    training_features, testing_features, training_target, testing_target = 
                train_test_split(features, tpot_data['fare_amount'], random_state=42)


# In[ ]:


exported_pipeline = make_pipeline(
        StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.9500000000000001, min_samples_leaf=10, min_samples_split=16, n_estimators=100))
        ElasticNetCV(l1_ratio=0.45, tol=0.001)\n",
    
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 42)
    
    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)


# In[ ]:


results


# In[ ]:


training_target


# In[ ]:


import seaborn as sns
sns.distplot(testing_target.values-results)


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(testing_target,results)


# In[ ]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(testing_target, results))
print('MSE:', metrics.mean_squared_error(testing_target, results))
print('RMSE:', np.sqrt(metrics.mean_squared_error(testing_target, results)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




