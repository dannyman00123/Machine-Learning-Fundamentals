import tensorflow
import keras
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

header = ["buying", "maintenance", "doors", "persons", "boot", "safety", "class"]
data = pd.read_csv("car.data", names=header, header=None)

#Preprocess file
replacer = preprocessing.LabelEncoder()
#Create lists of of our fields
buying = replacer.fit_transform(list(data['buying']))
maintenance = replacer.fit_transform(list(data['maintenance']))
door = replacer.fit_transform(list(data['doors']))
persons = replacer.fit_transform(list(data['persons']))
boot = replacer.fit_transform(list(data['boot']))
safety = replacer.fit_transform(list(data['safety']))
cls = replacer.fit_transform(list(data['class']))

predict = "class"

x = list(zip(buying,maintenance,door,persons,boot,safety))
y = list(cls)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


"""
best_score = 0
for i in range(1, 51, 2):
    for x in range(100):
        #x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_df, y_df, test_size=0.1)
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        if acc > best_score:
            best_score = acc
            print(str(acc) + "is higher than " + str(best_score))
            with open("cardatamodel" + str(i) + "_" + str(x) + ".pickle", 'wb') as f:
                pickle.dump(model, f)
"""

pickle_in = open("cardatamodel.pickle", 'rb')
model = pickle.load(pickle_in)

acc = model.score(x_test, y_test)
print("For this test data, accuracy of current model:", acc)
print("Optimal Neighbours for this model and this set was determined as ", str(model.n_neighbors))


predictions = model.predict(x_test)

missed_counter = 0
for x in range(len(predictions)):
    if predictions[x] != y_test[x]:
        missed_counter += 1

print("Out of the ", str(len(predictions)), "data points in the test set, ", missed_counter, "were incorrect")

# Fetch the data points of the data points we are predicting.

for x in range(0,1,1):
    n = model.kneighbors([x_test[x]], 7, False)
    print(x_test[x], predictions[x], y_test[x])
    print("N:", n)

# p = 'absences'
# style.use("ggplot")
# pyplot.scatter(data[p],data['failures'])
# pyplot.xlabel(p)
# pyplot.ylabel("Final Grade")
# pyplot.show()