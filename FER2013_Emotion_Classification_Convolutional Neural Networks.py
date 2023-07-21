# -*- coding: utf-8 -*-
"""
Emotion classification Using Convolutional Neural Networks with FER 2013 dataset By Shekhar Singh
# Read Dataset
# Initialize train and test dataset
# Data transformation for train and test datasets
# Batch process
# Load pre-trained weights
# Function for emotion preditions
# Plot Confusion Matrix
# Compute TP, TN, FP, FN
# Classification Report
# Make prediction for custom image
# Display Sample Predictions
# Display one misclassified image from each class
# Plot the class-wise accuracies

"""
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# variables
num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 64
epochs = 100

# Read Dataset
with open("FER 2013 Dataset or FER 2013 Augmented Dataset.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("number of instances: ",num_of_instances)
print("instance length: ",len(lines[1].split(",")[1].split(" ")))

#------------------------------
# initialize train and test dataset
x_train, y_train, x_test, y_test = [], [], [], []

#------------------------------

#transfer train and test set data
for i in range(1,num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
          
        val = img.split(" ")
            
        pixels = np.array(val, 'float32')
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
    
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
     print("",end="")

#------------------------------
#data transformation for train and test datasets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#batch process
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
test_generator= gen.flow(x_test, y_test, batch_size=batch_size)
#------------------------------

# Load pre-trained weights

model = tf.keras.models.load_model('savemodelandweights.hdf5')

"""
# Plot the Learning Curves using Model

fit = True
if fit:
    history = model.fit(x=train_generator, epochs=epochs, validation_data=test_generator)
else:
    model.load_weights('savemodelandweights.hdf5')  # Load weights
    # Alternatively, you can load the history from a saved file if available
    # history = load_history_from_file('path/to/history_file')

if fit:
    # Accuracy plots
    accuracy = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(accuracy))

    plt.plot(epochs_range, accuracy, 'r', label='Training accuracy')
    plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    plt.plot(epochs_range, loss, 'r', label='Training loss')
    plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc=0)
    plt.show()
"""
# Evaluate model
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc = model.evaluate(test_generator)
print("Final train accuracy = {:.2f}, validation accuracy = {:.2f}".format(train_acc * 100, test_acc * 100))


# function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()

#------------------------------

# def plot_confusion_matrix(actual_list, pred_list, classes,normalize=False,title=None,cmap=plt.cm.Blues):
predictions = model.predict(x_test) 
pred_list = []; actual_list = []
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
title='Confusion matrix'
for i in predictions:
    pred_list.append(np.argmax(i))
    #print(len(pred_list))
 
for i in y_test:
    actual_list.append(np.argmax(i))
    #print(len(actual_list))
 
cm=confusion_matrix(actual_list, pred_list)
# print(cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=90)
plt.yticks(tick_marks, labels)
fmt = 'd'
thresh = cm.max() / 2.
import itertools
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show() 
#------------------------------

# Compute TP, TN, FP, FN
TP = np.diag(cm)
TN = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + TP
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP

# Print TP, TN, FP, FN
for i in range(num_classes):
    print("Class: {}, TP: {}, TN: {}, FP: {}, FN: {}".format(i, TP[i], TN[i], FP[i], FN[i]))

# Classification Report
target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print(classification_report(actual_list, pred_list, target_names=target_names))


monitor_testset_results = False

if monitor_testset_results == True:
	#make predictions for test set
	predictions = model.predict(x_test)

	index = 0
	for i in predictions:
		if index < 30 and index >= 20:
			print(i) #predicted scores
			print(y_test[index]) #actual scores
			
			testing_img = np.array(x_test[index], 'float32')
			testing_img = testing_img.reshape([48, 48]);
			
			plt.gray()
			plt.imshow(testing_img)
			plt.show()
			
			print(i)
			
			emotion_analysis(i)
			print("----------------------------------------------")
		index = index + 1

#------------------------------

# Make prediction for custom image out of test dataset

img = image.load_img("FER 2013 Source_Images.png", color_mode="grayscale", target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(x)
plt.show()
#------------------------------

# Sample Predictions
num_samples = 5
sample_indices = np.random.choice(len(x_test), num_samples, replace=False)
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]

# Generate predictions for sample images
sample_predictions = model.predict(sample_images)
sample_pred_labels = np.argmax(sample_predictions, axis=1)

# Display sample images with predictions and actual labels
for i in range(num_samples):
    plt.imshow(sample_images[i].reshape(48, 48), cmap='gray')
    plt.title(f"Predicted: {target_names[sample_pred_labels[i]]}, Actual: {target_names[np.argmax(sample_labels[i])]}")
    plt.axis('off')
    plt.show()
    
print("Here are the misclassified images")

# Create a list to store the misclassified indices
misclassified_indices = []

# Iterate over the test set and find misclassified images
for i in range(len(x_test)):
    img = x_test[i]
    true_label = np.argmax(y_test[i])
    predicted_label = np.argmax(model.predict(np.expand_dims(img, axis=0)))
    
    if predicted_label != true_label:
        misclassified_indices.append(i)
    
    # Break if at least one misclassified image is found for each class
    if len(misclassified_indices) == num_classes:
        break
# Display one misclassified image from each class
for true_label in range(num_classes):
    misclassified_index = misclassified_indices[true_label]
    img = x_test[misclassified_index]
    predicted_label = np.argmax(model.predict(np.expand_dims(img, axis=0)))
    
    # Exclude the true label from the possible predicted labels
    possible_predicted_labels = [label for label in range(num_classes) if label != true_label]
    
    # Find a predicted label different from the true label
    for label in possible_predicted_labels:
        if label != true_label:
            predicted_label = label
            break
    
    plt.imshow(img.reshape(48, 48), cmap='gray')
    plt.title(f"Predicted: {target_names[predicted_label]}, Actual: {target_names[true_label]}")
    plt.axis('off')
    plt.show()

# Convert actual_list and pred_list to integer arrays
actual_list = np.array(actual_list)
pred_list = np.array(pred_list)

# Calculate class-wise accuracies
class_wise_accuracy = {}
for class_index in range(num_classes):
    class_samples = np.where(actual_list == class_index)[0]
    class_correct_samples = np.where(pred_list[class_samples] == class_index)[0]
    accuracy = len(class_correct_samples) / len(class_samples)
    class_wise_accuracy[class_index] = accuracy

# Plot the class-wise accuracies
plt.bar(class_wise_accuracy.keys(), class_wise_accuracy.values())
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Class-wise Accuracy')
plt.xticks(list(class_wise_accuracy.keys()))
plt.show()

"""    
# Below code is for the 7 misclassified images from each class

# Create a list to store misclassified indices for each class
misclassified_indices_by_class = [[] for _ in range(num_classes)]

# Iterate over the test set and find misclassified images
for i in range(len(x_test)):
    img = x_test[i]
    true_label = np.argmax(y_test[i])
    predicted_label = np.argmax(model.predict(np.expand_dims(img, axis=0)))
    
    if predicted_label != true_label:
        misclassified_indices_by_class[true_label].append(i)
    
    # Break if at least 7 misclassified images are found for each class
    if all(len(indices) >= 7 for indices in misclassified_indices_by_class):
        break

# Display 7 misclassified images from each class
for true_label, misclassified_indices in enumerate(misclassified_indices_by_class):
    print(f"Class: {target_names[true_label]}")
    for i in range(7):
        index = misclassified_indices[i]
        img = x_test[index]
        predicted_label = np.argmax(model.predict(np.expand_dims(img, axis=0)))
        
        plt.imshow(img.reshape(48, 48), cmap='gray')
        plt.title(f"Predicted: {target_names[predicted_label]}, Actual: {target_names[true_label]}")
        plt.axis('off')
        plt.show()
"""