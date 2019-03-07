# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 18:07:05 2018

@author: Xuan Liu
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

model = load_model('model.h5')

# load the text file
path = "E:/Graduate Study Material/ML/Project/modelPredict.txt"
modelPredict = np.genfromtxt(path , delimiter = None)
y_pred = modelPredict[:,1]   
y_true = modelPredict[:,2]

# plot the confusion matrix of the model
cm = confusion_matrix(y_true, y_pred)
emotion = ['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
cm_nor = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cm_nor, index = [i for i in emotion], columns = [i for i in emotion])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
plt.xlabel('predicted class')
plt.ylabel('original class')
plt.show()