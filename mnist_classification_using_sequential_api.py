import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt

mnist=keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)

# Normalising the data 

x_train = x_train/255.0
x_test = x_test/255.0
"""
#If you want to view how the training data looks like 
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
plt.show()
"""
# Building the model using sequential api
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # This will flatten our trainig images of shape(60000,28,28) to 47040000
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

print(model.summary())

#Another way to add models is 
#model = keras.Sequential()
#model.add(keras.layers.Fkatten(input_shape=(28,28)))
#model.add(keras.layers.Dense(128,activation='relu))
#model.add(keras.layers.Dense(10))


# Creating a loss and optimizer 
#loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) # from_logits=True is used if we have not added softmax function in the output layer
loss = keras.losses.SparseCategoricalCrossentropy()
optim = keras.optimizers.Adam(lr=0.001)
metric = ["accuracy"]

# Configuring the model for training
model.compile(loss=loss,optimizer=optim,metrics=metric)

# Training
batch=64
epoch=5
model.fit(x_train,y_train,batch_size=batch,epochs=epoch,shuffle=True,verbose=2)

# Evaluate 
model.evaluate(x_test,y_test,batch_size=batch,verbose=2)

# Prediction 
# Since we have not added  softmax in the final/output layer, so we are going to create a new model and we are goiing to pass the older model into that and also add a layer with softmax
#prob_model = keras.models.Sequential([ # Use if we have not included the softmax in our output layer
#    model,
#   keras.layers.Softmax() 
#]) 
print("\n")
predictions = model(x_test)
print("The predicted otuput is\n")
print(predictions,"\n") # The shape of this tensor is (10000, 10) because the test set contains the 10000 images and 10 is the number of class
#print(x_test.shape,"\n")
pred0= predictions[0]
print(type(pred0),"\n")
print(pred0,"\n")
label0 = np.argmax(pred0) # To know more about argmax visit https://machinelearningmastery.com/argmax-in-machine-learning/
print(label0)

print("\n\n")
predictions = model(x_test)
xx = len(predictions)
for i in range(xx):
    pred = predictions[i]
    #print(pred)
    label = np.argmax(pred)
    print(label)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[label], cmap='gray')
plt.show()