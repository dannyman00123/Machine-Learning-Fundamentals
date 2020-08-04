import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=';')

data = data[['G1','G2','G3','studytime','failures','absences']]
#print(data.head())


predict = "G3"
"""
Predict is whats known as a 'label'. In ML, a label is what you're trying to get.
A model may include multiple labels
"""

# Define two arrays, one being our attributes, and one being our label
x = np.array(data.drop([predict], 1)) # Everything but G3
y = np.array(data[predict]) # Only G3, both now in the form of a numpy array.

best_score = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""
for i in range(100000):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)

    acc = linear.score(x_test, y_test)
    #print(acc)
    #print(linear.coef_)
    #print(linear.intercept_)
    if acc > best_score:
        print(str(acc) + " is higher than " + str(best_score))
        best_score = acc
        with open("studentmodel.pickle", mode='wb') as f:
            pickle.dump(linear, f)

    pickle_in = open("studentmodel.pickle", 'rb')
    linear = pickle.load(pickle_in)
    
    
    predictions = linear.predict(x_test)
    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])
    
    # Saving our model - no point saving our model as it runs in under a second. We dont want to  re-run our model every single time

"""
#f.close()

pickle_in = open("studentmodel.pickle", 'rb')
linear = pickle.load(pickle_in)
acc = linear.score(x_test, y_test)
print(acc)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

print("hold")

#Plot results
p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p],data['failures'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()






