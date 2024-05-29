import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras #
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# print(X_train.shape,X_test.shape)
class_labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# plt.imshow(X_train[0],cmap='Greys')
j=1
plt.figure(figsize=(16,16))
for i in np.random.randint(0,1000,25):
     plt.subplot(5,5,j)
     j+=1
     plt.imshow(X_train[i],cmap='Greys')
     plt.axis('off')
     plt.title('{} / {}'.format(class_labels[y_train[i]],y_train[i]))
X_train= np.expand_dims(X_train,-1)
X_test= np.expand_dims(X_test,-1)
X_train=X_train/255
X_test=X_test/255
# Spliting of Dataset into Train and Test
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size=0.2,random_state=2020)
# model building
model=keras.models.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=3,strides=(1,1),padding='valid',activation='relu',input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
model.summary()
# Model Training
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,batch_size=512,verbose=1,validation_data=(X_validation,y_validation))
# Prediction
y_pred=model.predict(X_test)
y_pred.round(2)
model.evaluate(X_test,y_test)
j=1
plt.figure(figsize=(16,16))
for i in np.random.randint(0,1000,25):
    plt.subplot(5,5,j)
    j+=1
    plt.imshow(X_test[i],cmap='Greys')
    plt.axis('off')
    plt.title('actual= {} / {} \n predicted= {} / {}'.format(class_labels[y_test[i]],y_test[i],class_labels[np.argmax(y_pred[i])],np.argmax(y_pred[i])))
j=1
plt.figure(figsize=(16,30))
for i in np.random.randint(0,1000,60):
    plt.subplot(10,6,j)
    j+=1
    plt.imshow(X_test[i].reshape(28,28),cmap='Greys')
    plt.axis('off')
    plt.title('actual= {} / {} \n predicted= {} / {}'.format(class_labels[y_test[i]],y_test[i],class_labels[np.argmax(y_pred[i])],np.argmax(y_pred[i])))
# Confusion Matrix
plt.figure(figsize=(16,9))
y_pred_labels=[np.argmax(i) for i in y_pred]
cm= confusion_matrix(y_test,y_pred_labels)
sns.heatmap(cm,annot=True,fmt='d',xticklabels=class_labels,yticklabels=class_labels)
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred_labels,target_names=class_labels)
print(cr)
