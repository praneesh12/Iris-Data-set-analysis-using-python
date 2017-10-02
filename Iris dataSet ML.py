#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:14:38 2017

@author: praneeshkhanna
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


iris = load_iris()
#we will remove one example from each type of flower from the 
#training data set and put it in test data set : test_data.
test_idx = [0,50,100]

#iterate over 150 such enteries to print out the data set
#for i in range(len(iris.target)):
#    print("Example %d: label %s, feature %s" % (i+1, iris.target[i], iris.data[i]))

#traning data set
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#DecisionTree Classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))


#Predicting label for new flower
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     impurity = False)  
graph = pydot.graph_from_dot_data(dot_data.get_value)  
graph.write_pdf("iris.pdf")
 
    
