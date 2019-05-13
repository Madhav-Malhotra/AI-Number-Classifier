#Importing libraries needed
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO);

def NetCNN(features, labels, mode):
  """
  Parameters - 
  Feature map tensor
  Labels for handwritten digits
  EVAL, PREDICT, or TRAIN mode for CNN
  
  Output - 
  Returns predictions for each image in PREDICT mode
  Returns predictions with accuracy test results in
  EVAL mode

  """
  # Input Layer
  #Feature map is reshaped as input for grayscale
  #28 x 28 images with adjustable batch size.
  inputLayer = tf.reshape(features["x"], [-1, 28, 28, 1]);

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      #Layer connects to input
      inputs = inputLayer,
      #Layer applies 32 filters (find central value
      #of patch of pixels).
      filters = 32,
      #Each pixel patch is 5 by 5.
      kernel_size = [5, 5],
      #Preserve the size of the output tensor (28 x 28)
      padding = "same",
      #Activation function is Rectified Linear Unit
      activation = tf.nn.relu);

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(
    #Layer connects to first convolutional layer
    #Note: Tensor changed to 32 channels
    # [-1, 28, 28, 32]
    inputs = conv1, 
    #Size of max pooling filter
    pool_size = [2, 2],
    #Each subregion in filter has 2 x 2 border separation. 
    strides = [2, 2]);

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      #Layer connects to first pooling layer
      #Note: Tensor dimensions 50% smaller
      # [-1, 14, 14, 32]
      inputs = pool1,
      #Now applies 64 5x5 filters
      filters = 64,
      kernel_size = [5, 5],
      #Maintains same tensor shape as pool1
      padding = "same",
      activation = tf.nn.relu);
  
  #Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(
    #Layer connects to second covolutional layer
    #Note: Tensor changed to 64 channels 
    # [-1, 14, 14, 64]
    inputs = conv2, 
    pool_size = [2, 2], 
    strides = 2);

  # Dense Layer
  #Flattens pool2 2D tensor [-1, 7, 7, 64]
  #to 3136 long features dimension
  pool2Flat = tf.reshape(pool2, [-1, 7 * 7 * 64]);
  #Note: Batch size automatically calculated based
  #on number of input examples
  dense = tf.layers.dense(
    #Connects to flattened feature map
    inputs = pool2Flat,
    #1024-neuron layer with ReLU activation 
    units = 1024, 
    activation = tf.nn.relu);

  dropout = tf.layers.dropout(
      #Connects to dense neuron layer
      #Note: Tensor size is [-1, 1024]
      inputs = dense, 
      #40% of elements will be randomly dropped out 
      #to prevent overfitting
      rate = 0.4, 
      #Perform dropout if CNN is training
      training = mode == tf.estimator.ModeKeys.TRAIN);

  # Logits Layer - Returns raw values for predictions
  #Dense layer with 10 neurons for each digit
  #Note: Tensor size [-1, 10]
  logits = tf.layers.dense(inputs = dropout, units = 10);

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      #Records digit with highest predicted probability
      "classes": tf.argmax(input = logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      #Records predicted probability for all digits
      "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
  };

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions);

  # Calculate Loss (for both TRAIN and EVAL modes)
  #Uses softmax cross-entropy loss function
  loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits);

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    #Use a SGD optimisation with learning rate of 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001);
    train_op = optimizer.minimize(
        loss = loss,
        global_step = tf.train.get_global_step());
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op);

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels = labels, predictions = predictions["classes"])
  };

  return tf.estimator.EstimatorSpec(
      mode = mode, loss = loss, eval_metric_ops = eval_metric_ops);

# Load training and evaluation data
((trainData, trainLabels),
 (evalData, evalLabels)) = tf.keras.datasets.mnist.load_data();

#Store data as numpy arrays
trainData = trainData/np.float32(255)
trainLabels = trainLabels.astype(np.int32)  # not required
evalData = evalData/np.float32(255)
evalLabels = evalLabels.astype(np.int32)  # not required

print("Loaded dataset");

# Create the Estimator
mnistClassifier = tf.estimator.Estimator(
    #Uses CNN model function
    model_fn = NetCNN,
    #Saves model data checkpoint to this directory 
    model_dir="/tmp/mnist_convnet_model");

# Set up logging for prediction progress
#Logs output tensor (with predictions)
tensors_to_log = {"probabilities": "softmax_tensor"}
#Logs output tensor after every 50 training steps
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter = 50);

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #Inputting training feature data
    x = {"x": trainData},
    #Inputting training labels
    y = trainLabels,
    #Train on 100-example minibatches at a time
    batch_size=100,
    #Keeps running until specified steps completed
    num_epochs = None,
    #Shuffles training data
    shuffle = True);

# Trains Net 1000 Steps
mnistClassifier.train(input_fn = train_input_fn, steps = 1000);

#Evaluates prediction over 1 epoch of data in order.
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": evalData},
    y = evalLabels,
    num_epochs = 1,
    shuffle = False);

#Displays final accuracy results
evalResults = mnistClassifier.evaluate(input_fn = eval_input_fn);
print(evalResults);