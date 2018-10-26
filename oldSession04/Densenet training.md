# Densenet training
This has more than 1million parameters, which was one of the criteria for assignment 4. Hence giving up on this architecture and starting a new training in session04 folder

### Training 1:

##### notebook: 1Densenet_CIFAR10.ipynb

##### weights: 1DenseNet_model_24x24.h5 

##### Training time: 5 hours for 1 epoch!

Image resized to 28x28x3 (original size 32x32x3)

Hyperparameters:

batch_size = 64
num_classes = 10
epochs = 1
l = 20
num_filter = 64
compression = 0.75
dropout_rate = 0.2

```
12500/12500 [==============================] - 18262s 1s/step - loss: 0.8792 - acc: 0.6877
```

### Training 2:

##### notebook: 2Densenet_CIFAR10.ipynb

##### weights: 2DenseNet_model_24x24.h5 

##### Training time: 5 hours for 1 epoch! after one breakdown due to internet disconnect:(

Image resized to 28x28x3 (original size 32x32x3)

Hyperparameters:

batch_size = 64
num_classes = 10
epochs = 1
l = 20
num_filter = 64
compression = 0.75
dropout_rate = 0.2

```
12500/12500 [==============================] - 17489s 1s/step - loss: 0.4003 - acc: 0.8598 - val_loss: 14.5047 - val_acc: 0.1001
```

### Training 3:

##### notebook: 3Densenet_CIFAR10.ipynb

##### weights: 3DenseNet_model_24x24.h5 

Because it was taking too long for 1 epoch, changed the batch_size from 64 to 128. This caused OOM. Changed steps_per_epoch=len(x_train_28x28)/8 in model.fit_generator. It was earlier steps_per_epoch=len(x_train_28x28)/4. Still caused OOM. 

Changed back batch_size to 64 and increased dropout from 0.2 to 0.5. Accuracy reduced from 0.85 to 0.75 as the epoch began. But no OOM. But training time reduced as the number of images in each epoch was reduced due to change in steps_per_epoch(and dropout increase, not sure about dropout reducing training time though). Continuing with this setting even with reduced accuracy.... hoping that it would reduce overfitting and generalizes better on training data.

