# AI-Number-Classifier

This is a neural network to classify handwritten digits. 
I train the network with the MNIST Database and base the code from the book "Neural Networks and Deep Learning" by Michael A. Nielsen. 
The file mnist_loader.py is created by Michal Daniel Dobrzanski and can be found in this repository: 
https://github.com/MichalDanielDobrzanski/DeepLearningPython35.
The data used can be found here: 
http://yann.lecun.com/exdb/mnist/.
A tutorial to do this yourself can be found here:
http://neuralnetworksanddeeplearning.com/chap1.html.
It is my first project with Artificial Intelligence and uses fairly simple alogorithms.
I have not optimised the project and thus, performance varies from computer to computer. 

To Run the Program:
------------------------------------------------
Download the main.py, mnist_loader.py, and mnist.pkl.gz files into one directory.
Open the main.py file in a Python IDE (Editor). I would recommend https://repl.it as it is conveniently online.
In the Python shell, input the following:

import mnist_loader
trainingData, validationData, testingData = \
mnist_loader.load_data_wrapper()
import main
net = Network([784, 30, 10])
net.gradientDescent(trainingData, 5, 10, 3.0, \
testingData = testingData)
