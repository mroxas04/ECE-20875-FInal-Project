#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=0.4, shuffle=False)

def dataset_searcher(number_list,images,labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)
  images_nparray = []
  labels_nparray = []
  for number in number_list:
    search = 0
    while not(labels[search] == number):
      search += 1
    images_nparray.append(images[search])
    labels_nparray.append(labels[search])
  return images_nparray, labels_nparray
  #return images_nparray, labels_nparray

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 
  numList = labels
  fig, axs = plt.subplots(1, len(numList), figsize=(15,5))
  for index in range(len(numList)):
    # Display the image corresponding to the label
    axs[index].imshow(images[index], cmap="gray")
    axs[index].set_title("Label: " + str(numList[index]))
  plt.show()
class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images , class_number_labels )


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
model1_results = model_1.predict(X_test_reshaped) #What should go in here? Hint, look at documentation and some reshaping may need to be done)


def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  numCorrect = 0
  for x in range(len(results)):
    if results[x] == actual_values[x]:
      numCorrect += 1
  Accuracy = numCorrect / len(results)
  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
allnumbers_images = np.array(allnumbers_images)
allNums_reshaped = allnumbers_images.reshape(allnumbers_images.shape[0], -1)
allnumbersPrediction = model_1.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)

#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model_2results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model_2results, y_test)
print("The overall results of the K-Means model is " + str(Model2_Overall_Accuracy))
allnumbersPrediction = model_2.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)
#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
model_3results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model_3results, y_test)
print("The overall results of the MLP model is " + str(Model3_Overall_Accuracy))
allnumbersPrediction = model_3.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)


#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison
X_train_poison = X_train_poison.reshape(X_train.shape[0], -1)


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
model_1.fit(X_train_poison, y_train)
model_1results = model_1.predict(X_test_reshaped)
Model1_Overall_Accuracy = OverallAccuracy(model_1results, y_test)
print("The overall results of the Gaussian model with noise is " + str(Model1_Overall_Accuracy))
allnumbersPrediction = model_1.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)

model_2.fit(X_train_poison, y_train)
model_2results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model_2results, y_test)
print("The overall results of the KMeans model with noise is " + str(Model2_Overall_Accuracy))
allnumbersPrediction = model_2.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)

model_3.fit(X_train_poison, y_train)
model_3results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model_3results, y_test)
print("The overall results of the MLP model with noise is " + str(Model3_Overall_Accuracy))
allnumbersPrediction = model_3.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)
#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64
X_train_poison = np.reshape(X_train_poison, (len(X_train_poison), 64))
pca = PCA(n_components=2)
kernel_pca = KernelPCA(
    n_components=None,
    kernel="rbf",
    gamma=10,
    fit_inverse_transform=True,
    alpha=0.1,
    #random_state=42,
)
kernel_pca.fit(X_train_poison)
X_train_denoised = kernel_pca.inverse_transform(
    kernel_pca.transform(X_train_poison)
)


#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.
model_1.fit(X_train_denoised, y_train)
model_1results = model_1.predict(X_test_reshaped)
Model1_Overall_Accuracy = OverallAccuracy(model_1results, y_test)
print("The overall results of the Gaussian model with denoising is " + str(Model1_Overall_Accuracy))
allnumbersPrediction = model_1.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)

model_2.fit(X_train_denoised, y_train)
model_2results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model_2results, y_test)
print("The overall results of the KMeans model with denoising is " + str(Model2_Overall_Accuracy))
allnumbersPrediction = model_2.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)

model_3.fit(X_train_denoised, y_train)
model_3results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model_3results, y_test)
print("The overall results of the MLP model with denoising is " + str(Model3_Overall_Accuracy))
allnumbersPrediction = model_3.predict(allNums_reshaped)
print_numbers(allnumbers_images, allnumbersPrediction)