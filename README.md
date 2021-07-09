# Handwriting Reader - Java

A simple digit reading program written in Java. Utilises Processing for the GUI and a custom made Multi-Layer-Perceptron for the classification.

## Notes

+ The model has been trained on the MNIST dataset, specifically the modified version found at https://www.kaggle.com/oddrationale/mnist-in-csv
+ The model has an input layer of size 784, two hidden layers of size 400 and 205 respectively, and an output layer of size 10
+ The model utilises the sigmoid function as its activation function

## To Run

+ Ensure Gradle is installed on the computer
+ Download https://www.kaggle.com/oddrationale/mnist-in-csv and move the two csv files to src/main/resources
+ To train the model, run `gradle run --args="train"` (note that weights with 97% accuracy are included in the repo)
+ To run the program, run `gradle run`