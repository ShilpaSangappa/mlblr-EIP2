# DenseNet Training

The image is resized to 28x28 and is trained initially. After  3 epochs, the image is resized to 32x32 and trained again.

### Trainable parameters in the network: 

```Total params: 902,050
Total params: 902,050
Trainable params: 887,500
Non-trainable params: 14,550
```

### Total Number of epochs: 91

### Hyperparameters:

These are the hyperparameters unchanged throughout for all epochs:

batch_size = 128
num_classes = 10
l = 14
num_filter = 20
compression = 0.75
dropout_rate = 0.2

### Final Validation: 

https://colab.research.google.com/drive/10Ui4Ug7OdWOT09TAhRbYU5Bp-GVoePSS



Training has been conducted in different stages, with changes and tweaks. Following is the order of training with details of changes performed  and accuracy achieved at each stage.

### Training 1:

Input Image size: 28x28x3

File: https://colab.research.google.com/drive/1dljjPX8mGk1C0im_d5guw04U98ajGFbO

Training Accuracy achieved: Around 0.7

Test Accuracy achieved: Around 0.1

Epochs: 1

Time taken: 5+ hours

Comments: 

The images are resized using scipy.misc.imresize.

The reason 1 epoch took 5+ hours of training is heavy image augmentation. But it gave good accuracy and generalization. The difference between test accuracy and training accuracy was not that wide.

```python
datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train_28x28,y_train, batch_size=batch_size), 
                    steps_per_epoch=len(x_train_28x28), nb_epoch=epochs, verbose=1, callbacks = callbacks_list, validation_data=(x_test_28x28, y_test))
```



### Training 2:

Input Image size: 28x28x3

File: https://colab.research.google.com/drive/1B61YEpvFGIMl_enbxBmW5V3fSf3wp5rK

Training Accuracy achieved:  0.8935

Test Accuracy achieved: 0.149

Epochs: 2

Time taken: 1.5 hours. Around 45 mins per epoch

Comments: 

Continuing with the same setup as training1 for second epoch broke in between after around 4 hours of training. One epoch took 5+ hours in training 1, with 50000 test images. To tackle this google colab breakdown, I reduced the steps_per_epoch to reduce the images that each epoch looks at from 50000 to 6250 and increased the epochs to 2. This way it took 45 minutes per epoch and each epoch weights are saved using callback. 

Because weights are saved at the end of each epoch, if a single epoch takes a long duration, the chances of loosing weights are high; One alternative is to use steps_per_epoch and reduce training images in each epoch, and increase the number of epochs. Callbacks make sure that weights are saved at each epoch.

```
model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size), 
                    steps_per_epoch=len(x_train)/8, nb_epoch=epochs, verbose=1, 						callbacks = callbacks_list, 
                    validation_data=(x_test, y_test))
```

Changed from steps_per_epoch=len(x_train)  to steps_per_epoch=len(x_train)/8

### Training 3:

Input Image size: 28x28x3

File: https://colab.research.google.com/drive/1jDetQ4q4o0uFEusSgh2ek5FoeP-UTGbQ

Training Accuracy achieved: 0.8920

Test Accuracy achieved: 0.8713

Epochs: 2

Time taken: Around 180seconds per epoch. 260seconds for 2 epochs

Comments: 

No Image Augmentation. Plain model.fit()

```
model.fit(x_train_28x28, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,callbacks = callbacks_list, 
                    validation_data=(x_test_28x28, y_test))
```

Till now, number of epochs in total is 5 and total training time is around 6.5 hours. Test Accuracy achieved is 0.8713. Image Augmentation, although takes a lot of time, but having it for 1-2 epochs helps in achieving good accuracy. Also the difference between test accuracy and training accuracy is meagre. Image augmentation seems to help in generalization.

### Training 4:

Input Image size: 32x32x3

File: https://colab.research.google.com/drive/1Y8TOk2qn_8ZcbkkmOsuPMZ4SCSJ-NEzG

Training Accuracy achieved: 0.9552

Test Accuracy achieved: 0.9014

Epochs: 30

Time taken: Around 260 seconds per epoch. 

2.15 hours for 30 epochs

Comments: 

The weights saved in training 3 cannot be directly loaded here, as the image resolution has been changed from 28x28 to 32x32. This is because number of parameters in FC layers is not the same with change in input image size. Hence we need to pop FC and flatten layer, load the weights saved during training 3, add the FC and flatten layer and perform training.

### Training 5:

Input Image size: 32x32x3

File: https://colab.research.google.com/drive/1POyJ6PkkwaEcfZW6Y7vxBDTJ3z4guGVu

Training Accuracy achieved: 0.9221

Test Accuracy achieved: 0.2154

Epochs: 3

Time taken: Around 4000 seconds per epoch. 

Around 3.5 hours for 3 epochs

Comments: 

Brought in image augmentation with the hope that it would increase accuracy.

```python
datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size), 
                    steps_per_epoch=len(x_train)/8, nb_epoch=epochs, verbose=1, 						callbacks = callbacks_list, 
                    validation_data=(x_test, y_test))
```



### Training 6:

Input Image size: 32x32x3

File: https://colab.research.google.com/drive/1eW0aHKsDmHMxIIQYBO9Mu_QWgir2-A_k

Training Accuracy achieved: 0.9584

Test Accuracy achieved: 0.9118

Epochs: 10

Time taken: Around 270 seconds per epoch. 

Around 45 minutes for 10 epochs

Comments:  Reduced the learning rate. lr=0.01 to lr=0.001



### Training 7:

Input Image size: 32x32x3

File: https://colab.research.google.com/drive/1rUCe1N7k-mbdtI2fmMZBodHADWubRRE8

Training Accuracy achieved: 0.9712

Test Accuracy achieved: 0.9166

Epochs: 30

Time taken: Around 270 seconds per epoch. 

Around 2.25 hours for 30 epochs

Comments:  Reduced the learning rate. lr=0.01 to lr=0.001



### Training 8:

Input Image size: 32x32x3

File: https://colab.research.google.com/drive/1DJBEz2-7qUOqZbnu-8uY0jxA6kbk9hfW

Training Accuracy achieved: 0.9710

Test Accuracy achieved: 0.9164

Epochs: 10

Time taken: Around 270 seconds per epoch. 

Around 45 minutes for 10 epochs

Comments:  Reduced the learning rate. lr=0.01 to lr=0.001

NO INCREASE IN ACCURACY



### Training 9:

Input Image size: 32x32x3

File: https://colab.research.google.com/drive/1rS-B0z7BAKdsIKkOZTZMm_PcaaVZbueh

Training Accuracy achieved: 0.9721

Test Accuracy achieved: 0.9171 (Epoch 2 had this accuracy, which was the highest. Taking weights from this 2nd epoch)

Epochs: 3

Time taken: Around 270 seconds per epoch. 

Around 13.5 minutes for 3 epochs

Comments:  Reduced the learning rate. lr=0.01 to lr=0.001

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)

Changed nesterov=False.



