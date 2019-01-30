#Getting initial data
import numpy
import random

'''
import mnist_loader
>>> training_data, validation_data, test_data = \
... mnist_loader.load_data_wrapper()
net = Network([784, 30, 10]);
net.gradientDescent(trainingData, 30, 10, 3.0);
'''

class Network (object):

  def __init__(self, neurons):
    '''
    Creates a neural network. Argument is a list of the neurons per layer.
    '''
    #Number of layers in neural network
    self.numLayers = len(neurons);
    #Number of neurons per layer.
    self.neurons = neurons;
    #Initialises random biases for all layers except input
    self.biases = [numpy.random.randn(y, 1) for y in neurons[1:]];
    #Initialises random weights for all layers except input
    self.weights = [numpy.random.randn(y, x) for x, y in zip(neurons[:-1], neurons[1:])];

  def feedforward(self, activation):
    '''
    Gives network's output with a certain input.
    One neuron's output goes to next and so on.
    '''
    for bias, weight in zip(self.biases, self.weights):
      activation = sigmoid(numpy.dot(weight, activation) + bias);
    return activation;
  
  def gradientDescent(self, trainingData, trainingRounds, sizeBatches, learnRate, testingData = None):
    '''
    trainingData = List of tuples with inputs marked with correct outputs. Format (result, desiredResult).

    trainingRounds = Number of times to repeat training. One round is every time all trainingData is used. Positive integer.

    sizeBatches = The size of mini batches data is broken up into. Positive integer.

    learnRate = How quickly network adjusts weights and biases.

    testingData = Optional. Tests neural network after each trainingRound for progress.
    '''
    #This defines the number of training cases.
    trainingData = list(trainingData);
    numTrain = len(trainingData);
    # This defines the number of test cases if testing data is provided.
    if testingData: 
      testingData = list(testingData);
      numTest = len(testingData);
    #This runs through the number of rounds asked.
    for currentRound in range(trainingRounds):
      #This shuffles the training data for each round.
      random.shuffle(trainingData);
      #This divides the training data into smaller batches based on the batch size provided.
      batches = [
        trainingData[startIndex:startIndex + sizeBatches] for startIndex in range(0, numTrain, sizeBatches)];
      #This updates the network's weights and biases for each batch tested.
      for currentBatch in batches:
        self.updateCurrentBatch(currentBatch, learnRate);
      #If there is a testing data, this outputs the network's accuracy per round.
      if testingData:
        print("Training round {} : {} / {}".format(currentRound, self.evaluate(testingData), numTest));
      #Otherwise, this outputs how many rounds have been done.
      else:
        print("Training round {} complete".format(
          currentRound));

  def updateCurrentBatch(self, currentBatch, learnRate):
    '''
    Updates network's weights and biases using backpropogation.

    currentBatch = list of tuples with training data. Format (result, desiredResult). 

    learnRate = How quickly network adjusts weights and biases.
    '''
    #This calculates derivatives and stuff.
    derivativeBiases = [numpy.zeros(bias.shape) for bias in self.biases];
    derivativeWeights = [numpy.zeros(weight.shape) for weight in self.weights];
    #Goes through network's outputs for training data. 
    for result, desiredResult in currentBatch:
      #Uses backpropogation to find adjustments required in weights and biases.
      changeBiases, changeWeights = self.backpropogation(result, desiredResult);
      #Calculates derivatives and stuff for biases.
      derivativeBiases = [db + cb for db, cb in zip(
        derivativeBiases, changeBiases)];
      #Calculates derivatives and stuff for weights.  
      derivativeWeights = [dw + cw for dw, cw in zip(
        derivativeWeights, changeWeights)];
      #Adjusts network's weights given learning rate.
      self.weights = [weight - (learnRate / len(currentBatch)) * derivativeWeight for weight,  derivativeWeight in zip(self.weights,   derivativeWeights)];
      #Adjusts network's biases given learning rate.
      self.biases = [bias - (learnRate / len(currentBatch)) * derivativeBias for bias, derivativeBias in zip(self.biases, derivativeBiases)];

  def backpropogation(self, result, desiredResult):
    '''
    Finds what adjustments are needed using gradient descent. Minimises cost.

    Returns tuple with gradient for cost function.
    Format (derivative biases, derivative weights).
    '''
    #This calculates derivatives and stuff.
    derivativeBiases = [numpy.zeros(bias.shape) for bias in self.biases];
    derivativeWeights = [numpy.zeros(weight.shape) for weight in self.weights];
    #Feeding values forward through Network
    activation = result;
    #Stores neuron activations for each layer.
    activations = [result];
    #Stores neuron outputs for each layer.
    outputs = [];
    #Calculates neuron outputs for each layer. 
    for bias, weight in zip(self.biases, self.weights):
      output = numpy.dot(weight, activation) + bias;
      outputs.append(output);
      activation = sigmoid(output);
      activations.append(activation);
    #Calculating adjustments necessary backwards through Network.
    change = self.costDerivative(activations[-1], desiredResult) * sigmoidPrime(outputs[-1]);
    derivativeBiases[-1] = change;
    derivativeWeights[-1] = numpy.dot(change, activations[-2].transpose());
    #Calculating necessary derivatives and stuff.
    for lastLayer in range(2, self.numLayers):
      output = outputs[-lastLayer];
      sigmoidDerivative = sigmoidPrime(output);
      change = numpy.dot(self.weights[-lastLayer + 1].transpose(), change) * sigmoidDerivative;
      derivativeBiases[-lastLayer] = change;
      derivativeWeights[-lastLayer] = numpy.dot(change, activations[-lastLayer - 1].transpose());
    #This returns the derivatives for adjustments.
    return (derivativeBiases, derivativeWeights);

  def evaluate(self, testingData):
    '''
    Returns number of tests with correct network output. Output is neural network with highest activation in output layer.
    '''
    #Gets test outuputs compared to correct outputs.
    testResults = [(numpy.argmax(self.feedforward(result)), desiredResult) for (result, desiredResult) in testingData];
    #Returns number of correct outputs.
    return sum(int(result == desiredResult) for (result, desiredResult) in testResults);

  def costDerivative(self, outputActivations, desiredResult):
    '''
    Returns partial derivatives and stuff for calculating how much network's weights and biases need to be adjusted.
    ''' 
    return (outputActivations - desiredResult);

def sigmoid(output):
    '''
    Applies the sigmoid function. Compresses input to range from 0 to 1.
    Output closer to 1 with more positive inputs. 
    Ouput closer 0 with more negative inputs.
    '''
    return 1.0 / (1.0 + numpy.exp(-output));

def sigmoidPrime(output):
    '''
    Derivative of the sigmoid function.
    '''
    return sigmoid(output) * (1 - sigmoid(output));