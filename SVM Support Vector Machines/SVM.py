import sklearn
from sklearn import datasets
from sklearn import svm
import matplotlib as mlp
from matplotlib import style
from matplotlib import pyplot
from sklearn import metrics



# Loading in a dataset about Cancer location

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

labels = ['malignant','benign']

model = svm.SVC()

model.fit(x_train,y_train)
predictions = model.predict(x_test)

acc = metrics.accuracy_score(y_test,predictions)
print(acc)

missed_counter = 0
for x in range(len(predictions)):
    #print(x_test[x], predictions[x], y_test[x])
    if predictions[x] != y_test[x]:
        missed_counter += 1

print("Out of the ", str(len(predictions)), "data points in the test set, ", missed_counter, "were incorrect",  missed_counter / len(predictions) * 100, "% Failure Rate")

# p = 'mean radius'
# style.use("ggplot")
# pyplot.scatter(data[p],data['failures'])
# pyplot.xlabel(p)
# pyplot.ylabel("Final Grade")
# pyplot.show()