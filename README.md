import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the data
input_data = (5,166,72,19,175,25.8,0.587,51)

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the StandardScaler object to the input data
scaler.fit(np.array(list(input_data)).reshape(-1, 1))

# Transform the input data using the StandardScaler object
std_data = scaler.transform(input_data.reshape(-1, 1))

# Create a RandomForestClassifier object
rf_model = RandomForestClassifier()

# Fit the RandomForestClassifier object to the training data
rf_model.fit(X_train, Y_train)

# Make predictions using the RandomForestClassifier object
prediction1 = classifier.predict(std_data)
prediction2 = rf_model.predict(X_test)
prediction3 = knn.predict(X_test)
prediction4 = gnb.predict(X_test)
prediction5 = model.predict(X_new)

# Print the predictions
print(prediction1)
print(prediction2)
print(prediction3)
print(prediction4)
print(prediction5)

# Check if the person is diabetic or not
if (prediction1[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
if (prediction2[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
if (prediction3[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
if (prediction4[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
if (prediction5[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
