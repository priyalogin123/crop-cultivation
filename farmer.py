#PROBLEM STATEMENT:   HELP OUR FARMERS

import pandas as pd
crop1 = pd.read_csv('crop1.csv')
crop1.head()

crop1.columns
cols_to_norm = ['Weather','crop','Month','Planting','Soiltype','Waterneed']
#crop1[cols_to_norm] = crop1[cols_to_norm]

import tensorflow as tf
weat = tf.feature_column.numeric_column('Weather')
crop = tf.feature_column.numeric_column('crop')
mon = tf.feature_column.numeric_column('Month')
plan = tf.feature_column.numeric_column('Planting')
soil = tf.feature_column.numeric_column('Soiltype')
water = tf.feature_column.numeric_column('Waterneed')

feat_cols = [weat,crop,mon,plan,soil,water]
cropname = input("Enter the crop name:")
soil = input("Enter the soil type:")


from sklearn.tree import DecisionTreeClassifier
x_data = crop1.drop('Month',axis=1)
labels = crop1['Month']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=14)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=14)
model.train(input_fn=input_func,steps=100)

for i in range(14):
    if cropname == crop1.crop[i]:
        print(crop1.Weather[i], crop1.Month[i])
     
       
