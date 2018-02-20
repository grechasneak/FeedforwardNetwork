# FeedforwardNetwork
This neural network was implemented to classify images in the CIFAR-10 dataset into two classes.
The network has one hidden layer with a ReLu activation function and a two class cross entropy cost function at the output.
The model was trained using stochastic mini-batch gradient descent with momentum optimization.

The model has four parameters that can be optimized: number of hidden units, learning rate,
mini-batch size, and momentum friction factor. There is a trade off between learning rate and minibatch
size, since with a larger mini batch size you reduce the variance of the stochastic gradient
updates, and thus are able to take a larger step size. The momentum affects how large the training
step is, and thus it is also balanced by the mini-batch size.
