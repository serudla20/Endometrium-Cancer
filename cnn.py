import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
base_dir="D:\ENDOMETRIUM_UPDATED\test"

class_labels=os.listdir(base_dir)
print(class_labels)

data=[]
count=0

for label in class_labels:
    i=0
    path = os.path.join(base_dir, label)
    print(path)
    for img in os.listdir(path):
        try:
            image=load_img(os.path.join(path, img), grayscale=True, color_mode='grayscale', target_size=(128,128))
            image=img_to_array(image)
            image=image/255.0
            data.append([image,count])
        except Exception as e:
                pass
    count=count+1

print(len(data))

X,y =zip(*data)

X=np.array(X)
y=np.array(y)

unique,counts=np.unique(y,return_counts=True)
chart_data=list(counts)
chart_labels=['EA','EH','EP','NE']
colors=['mediumspringgreen','mediumaquamarine','aquamarine','maroon']
with plt.style.context(style='fivethirtyeight'):
    plt.figure(figsize=(18,8))
    plt.rcParams['font.size']=18
    plt.bar(x=chart_labels,height=chart_data,color=colors)
    plt.title(label='Records counts of classes using bar-chart')
    plt.xlabel(xlabel='Labels')
    plt.ylabel(ylabel='Number of records')
    plt.show()

with plt.style.context(style='fivethirtyeight'):
    plt.figure(figsize=(5,5))
    plt.pie(x=chart_data,labels=chart_labels,colors=colors,autopct='%.2f%%',explode=None,startangle=90)
    plt.title(label='Records counts of classes using pie-chart')
    plt.show()

from random import randint
def show_images(X,y):
    labels =class_labels
    x,y = X,y
    plt.figure(figsize=(15, 15))
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        #idx = randint(0, 2890)
        idx = randint(0, 2790)
        plt.imshow(x[idx],cmap='gray')
        plt.axis("off")
        plt.title("Class:{}".format(labels[y[idx]]))
show_images(X,y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

y_train=to_categorical(y_train,dtype="int32")
y_test=to_categorical(y_test,dtype="int32")

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam

cnn_model=Sequential()

cnn_model.add(Conv2D(filters=64,kernel_size=3,strides=(2,2),padding="same",activation="relu",input_shape = (128,128,1)))
cnn_model.add(Conv2D(filters=64,kernel_size=3,strides=(2,2),padding="same",activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size=(3,3),padding="same"))

cnn_model.add(Conv2D(filters=128,kernel_size=3,strides=(2,2),padding="same",activation="relu"))
cnn_model.add(Conv2D(filters=128,kernel_size=3,strides=(2,2),padding="same",activation="relu"))
cnn_model.add(MaxPool2D(pool_size=(3,3),padding="same"))

cnn_model.add(Conv2D(filters=256,kernel_size=3,strides=(2,2),padding="same",activation="relu"))
cnn_model.add(Conv2D(filters=256,kernel_size=3,strides=(2,2),padding="same",activation="relu"))
cnn_model.add(MaxPool2D(pool_size=(3,3),padding="same"))

cnn_model.add(Flatten())
cnn_model.add(Dropout(rate=0.2))
cnn_model.add(Dense(units=512,activation="relu"))
cnn_model.add(Dense(units=4,activation="softmax"))

cnn_model.compile(optimizer=Adam(learning_rate=1e-4),loss="categorical_crossentropy",metrics=["accuracy"])

history=cnn_model.fit(x=X_train,y=y_train,batch_size=64,epochs=100,validation_data=(X_test,y_test))

plt.figure(figsize=(18,8))
plt.rcParams["font.size"]=15
plt.plot(history.history["accuracy"],label="train_accuracy")
plt.plot(history.history["val_accuracy"],label="val_accuracy")
plt.title(label="Training Accuracy and Val_accuracy plot-graphs")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("acc_graph.png")
plt.show()


plt.figure(figsize=(18,8))
plt.rcParams["font.size"]=15
plt.plot(history.history["loss"],label="train_loss")
plt.plot(history.history["val_loss"],label="val_loss")
plt.title(label="Training Losss and Val_loss plot-graphs")
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig("loss_graph.png")
plt.show()

cnn_prediction=cnn_model.predict(X_test,batch_size=64,verbose=1)

cnn_labels=[]
for i in range(len(cnn_prediction)):
    cnn_labels.append(np.argmax(cnn_prediction[i]))

true_labels=[]
for i in range(len(y_test)):
    true_labels.append(np.argmax(y_test[i]))

cnn_accuracy=accuracy_score(y_true=true_labels,y_pred=cnn_labels)
print("CNN Accuracy is {:.2f}%".format(cnn_accuracy*100.0))

from sklearn.metrics import classification_report
print(classification_report(y_true=true_labels,y_pred=cnn_labels,target_names=class_labels))

import mlxtend
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
ax=plot_confusion_matrix(conf_mat=confusion_matrix(y_true=true_labels,y_pred=cnn_labels),
                        figsize=(10,5),
                        class_names=class_labels,
                        cmap=plt.cm.Reds)
plt.title(label="CNN Confusion Matrix")
plt.xticks(rotation=90)
plt.savefig("conf_mat.png")
plt.show()


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained CNN model
cnn_model_path = "models/Convolutional_Neural_Network.h5"
cnn_model = load_model(cnn_model_path)

# Define the classes
class_labels = os.listdir(base_dir)

# Function to prepare the image for prediction
def prepare_test_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128), grayscale=True)
    x = image.img_to_array(img)
    x = x / 255.0
    return np.expand_dims(x, axis=0)

# Function to predict and display the result
def predict_and_display_image(model, img_path, class_labels):
    result = model.predict(prepare_test_image(img_path))
    img = cv2.imread(img_path)

    # Display the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Test Image')
    plt.show()

    print("Prediction Result:", result[0])

    class_result = np.argmax(result, axis=1)
    print("Class number:", class_result[0])
    print("Predicted class:", class_labels[class_result[0]])

# Test with an image
test_image_path = 'C:\\Users\\DELL\\Desktop\\Endometrium\\a.jpg'  # Replace with the actual path of your test image
predict_and_display_image(cnn_model, test_image_path, class_labels)
