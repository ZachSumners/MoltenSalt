import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

print(digits.images[0])
fig, axs = plt.subplots(1, 4, figsize=(10,3))
for i in range(4):
    axs[i].imshow(digits.images[i], cmap=plt.cm.gray_r)

data = digits.images.reshape(len(digits.images), -1)

classifier = svm.SVC(gamma = 0.001)

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5)

classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)

print(
    f"Classification report for classifier {classifier}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
