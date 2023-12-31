# hgalla-a4

## Assignment 4: Machine Learning

#### Part1: K-Nearest Neighbours classification:

For finding the k-nearest neighbours, I first implemented the two distance functions which will help us to get an idea of the proximity of training label to the test label. For implementing this and the rest of the things in the assignment, I used only the 'numpy' package, which tend to be more efficient and easy-to-use than normal python lists.

For the KNearestNeighbors we get input variables like k-value: k-neighbors which are to be the nearest, weights: which will tell us whether the samples are to be given a preference according to the distances or uniformly, metric like 'l1' or 'l2' which will indicate which distance metric to use: euclidean or manhattan.

First, I implemented the distance functions which we kind of the main thing in this part. The euclidean distance function take 2 vectors as input and it calculates the distance by comparing the difference between each dimension and takes the sum of those squares.

**sqrt((x1-y1)^2 + (x2-y2)^2 + ....)**

The manhattan is straightforward, it just returns the absolute value of the difference between the vectors:

|x1-y1| + |x2-y2| + ....

Next, in the fit stage I just store the input training data and output classes. The main operation occurs in the predict stage. When it is called with a test data, I iterate through all the samples and find the k samples with least distances. I then take a note of the class labels. Now if the distance is uniform I just return the majority, but if it is not then I assign weights to each of the samples inversionally proportional to the distances. Thus the sample with least distance will get the most weight. Based on this weighted result I return the most likely class prediction.

The outputs for this part, very good and my accuracies matched the ones from Sklearn model which was quite surprising to be honest at first. But given how simple the fundamental concept is behind this problem, it all made sense.

#### Part2: Multilayer Perceptron Classification:

Multilayer Perceptron Model is a feedforward neural network which consists of interconnected neurons. These neurons are fundamental units which perform simple calculations of the input along with weights and bias. To get started, first I implmented the functions in the util file which included activation functions like identity, sigmoid, relu etc. and their derivatives. In addition to that, I implemented the cross entropy loss which helps us estimate the gradient loss between output and training data to adjust the weights. For the output, I implemented one hot encoding, which is then fed to the neural network. For implementing all these functions, using numpy really came in handy. The main thing, I observed is that numpy arrays allow us to translate our mathematical formulas easily in form of code. I referred to wolframalpha to get the information about these functions and implement them easily using numpy arrays.

The MultilayerPerceptron object is initialised with parameters like number of hidden neurons, hidden_activation function to be used, no of iterations and the learning rate. In the init method we find the number of features and outputs. We then give some random inital weights and bias values for both the output and hidden layers.

When the fit method is called with the training data, initial inputs and outputs, the main training action will occur. The input will first be fed in a feed forward fashion to the hidden layer. I created a reusable function which will calculate the neuron output when the initial inputs,weights and bias values are passed to it along with the activation function. The same thing repeats from the hidden layer to the output layer which will give us some an output.

Once we get the output, I compare it calculate the gradient loss cross entropy which will give me the correction values, according to which the weights and bias for both input and hidden layers are adjusted. Post this, the cycle repeats for the number of iterations given the learning rate mentioned. This will help properly train the model.

Now here, the predict part is straightforward. Given the input, we just compute its result by passing through the trained network and return the output. Although, I was able to get the output, I found it hard to debug and see where the it's going wrong. As I can observe from the results, my accurancies in some scenarios is very good and very few situtations better than the scikit model, however when the accuracies drop, they drop pretty hard and cannot figure out why. Overall, this was a good experience implementing a well-known algorithm from scratch.
