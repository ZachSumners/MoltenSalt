import matplotlib.pyplot as plt

test = [0.709854015,0.724452555,0.684306569,0.714808044,0.718464351]
train = [0.76347032,0.753881279,0.761643836,0.753537198,0.755819261]

plt.bar([0.8, 1.8, 2.8, 3.8, 4.8], test, width=0.4, label='Testing', color='dodgerblue')
plt.bar([1.2, 2.2, 3.2, 4.2, 5.2], train, width=0.4, label='Training', color='gray')
plt.legend()
plt.xlabel('Cross Validation Fold')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
plt.title('Support Vector Machine Cross Validation Performance')
plt.show()