# Assignment Session 1
##### Name:Shilpa Sangappa

##### Batch: 9

*------------------------------------------------------------------*

### Topic 1: Convolution

Convolution is the process of running a kernel/filter over an image to extract features. It basically is sum of product of image and kernel values.

Consider a binary image of size 3*3, with the following pixel values:

1 | 0 | 1
1 | 1 | 0
1 | 1 | 1

Consider a 3*3 kernel:
0 | 0 | 0
0 | 1 | 0
0 | 0 | 0

The output of convolving this kernel over the image is:

1 

### Topic 2: Kernels/Filters

Kernels or Filters are used to extract features by performing a convolution between the image(if we are dealing with images and computer vision) and the kernel.

Incase of images, a kernel is convolved over the whole image to extract  features. In a convolutional neural network, layers of convolution are done, so that the first layer picks up the most elementary features say edges, the second layer picks the second most elementary features like loops and lines,  and further up layers pick the predominant features.

The size of a kernel also has a huge impact. Small kernels pick up minute details, whereas large kernels discard quite a lot of features. The selection of kernel size depends on the application being developed. 

Convolving a 3*3 kernel  over an image reduces its size by two in both the dimensions.

The values of the kernel determine the kind of features we are interested in extracting. 

###### Kernel to extract point:  
-1 | -1 | -1
-1| 8 | -1
-1 | -1 | -1

###### Kernel to extract horizontal line:
-1 | -1 | -1
2  |  2 | 2
-1 | -1 | -1

###### Kernel to extract vertical line:
-1 | 2 | -1
-1 | 2 | -1
-1 | 2 | -1

In traditional image processing, kernels were created with specific values.
But in Convolutional Neural Networks, kernel values are randomly initiated and the network learns these kernel values during the process of training.

### Topic 3: Epochs

Epochs is the number of times an image is convolved on. 

The dataset is split up into training and test set. The training dataset images are used to train the CNN. To improve accuracy, the network has to be well-trained. Hence, the network is trained on the images over and over again. 

Epochs also is very helpful when dataset is small. Having greater number of epochs lets network see the images more and extract more features. But the drawback of it is overfitting. Hence, the number of epochs should be carefully evaluated.

### Topic 4: 1X1 Convolution

1x1 kernel is looking at just one pixel from a feature map. Hence it is not picking any features in the spatial domain. But it is picking up features by merging the different feature maps.

Adding layers one after the other makes the network grow massive and cause out of memory(OOM) issues. A 1X1 kernel reduces number of feature maps.

A 1x1 kernel also helps in only passing the features of objects relevant to us and shuns the background information.

### Topic 5: 3X3 Convolution

3x3 convolution is convolving the image or feature map with a kernel of size 3x3. It is extensively used as it is computationally efficient. Convolution using a single 5x5 kernel is equal to performing 3x3 convolution twice. But the number of parameters and computations involved in 3x3 convolutions(9+9=18 multiplications) is lesser than a single 5x5 convolutions(25 multiplications)

### Topic 6: Feature Maps

The output of each layer in a CNN form what are called feature maps, which inturn is passed as an input to the next layer. 

The size of the output feature map depends on the size of the input image or input feature map and the size of the kernel. For example, if the input feature map is of size 400x400 and the kernel size is 3x3, then each feature map in the output of this layer would be of size 398x398.

The number of feature maps in the output of  a layer is equal to the number of kernels in the layer. If there are 32 kernels in a layer, then there will be 32 feature maps in the output of that layer.

### Topic 7: Feature Engineering

Feature Engineering is involving domain knowledge to decide how features can be used. 

In the case of images, pixel values are the features. If the size of an image is say 400x400, then having a fully connected layer would have 640000 inputs in the first layer. This is one approach to feature engineering. 

An alternative feature engineering method would be to use a CNN. Using a CNN with a 3x3 kernel clubs 9 pixels within an image and performs feature extraction, allowing us to use spatial information, which is lost when a fully connected layer is used.

### Additional Topic 1: Activation Functions

The model of a neuron in an Artificial Neural Network(ANN) is as shown in the below figure:

![Image](https://www.researchgate.net/profile/Rubem_Koide/publication/282683862/figure/fig3/AS:307649800359938@1450360836478/Nonlinear-model-of-a-neuron-Haykin-1999.png)

A neuron performs a sum of products of its inputs and their respective weights and to this applies an activation function, which will be the output of the neuron. 

The main purpose of an activation function is to introduce non-linearity. A neural network without any activation function is equivalent to linear regression. An activation function produces a non-linear decision boundary through a non-linear combination of weights. This makes them the so called "Universal Function Approximators".

An activation function also needs to be differentiable, so as to perform backpropogation for training the network.

#### Common activation functions:
###### 1. Sigmoid Function
![Image](https://cdn-images-1.medium.com/max/640/1*DHN75JRJ_EQgGc0spfqLtQ.png)
![Image](https://cdn-images-1.medium.com/max/640/0*5euYS7InCmDP08ir.)

With sigmoid function, the neuron output is restricted to the range of 0 and 1. This helps us keep the activations bound to a range, instead of it ranging from (-infinity) to (+infinity). But it's huge drawback is vanishing gradient.

###### 2. Tanh Function
![Image](https://cdn-images-1.medium.com/max/640/1*WNTLbBRWFiHPoXvyZ6s9eg.png)
![Image](https://cdn-images-1.medium.com/max/640/0*YJ27cYXmTAUFZc9Z.)

Tanh is a scaled sigmoid activation function. 
tanh(x) = 2*sigmoid(2x) - 1

Although it stretches the range of output to (-1,1), it still suffers from vanishing gradient.

###### 3. Rectified Linear Unit(ReLU)
A(x) = max(0,x)
![Image](https://cdn-images-1.medium.com/max/640/0*vGJq0cIuvTB9dvf5.)

ReLU is the most popular activation function in use today. It's range is (0,infinity). 
But because for the region where the input is less than zero, it's output is zero, it again introduces vanishing gradient problem in that region. To tackle this problem, variants of it Leaky ReLU, Randomized ReLU are in use.

### Additional Topic 2: Receptive Field

Receptive field is a way to describe how much of an image have we seen at each layer at a pixel level 

If the input image is of size 400x400, and we have two 3x3 convolution layers, the first convolution layer has a receptive field of 3x3. The second convolution layer has a receptive field of 5x5. The receptive field increases by 2x2 with each layer that we add, using 3x3 kernels

The number of layers in a network is decided based on the receptive field we wish to attain. 


### References
*______________________*

Digital Image Processing, Author - Rafael C. Gonzalez

[What is the Role of the Activation Function in a Neural Network?](https://www.kdnuggets.com/2016/08/role-activation-function-neural-network.html)

[Activation functions and it’s types-Which is better?](https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f)

[Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)


[math - Why must a nonlinear activation function be used in a backpropagation neural network? - Stack Overflow](https://stackoverflow.com/questions/9782071/why-must-a-nonlinear-activation-function-be-used-in-a-backpropagation-neural-net)

[Activation Functions: Neural Networks – Towards Data Science](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

[](http://www.bioinfo.org.cn/~casp/temp/DeepLearning.pdf)
