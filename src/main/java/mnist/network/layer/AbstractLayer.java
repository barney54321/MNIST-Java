package mnist.network.layer;

import mnist.network.Util;

import java.util.*;

public abstract class AbstractLayer implements Layer {

    protected int size;

    protected int inputSize;

    protected double[][] weights;

    protected double[] biases;

    private double[] lastOutput;

    protected Random rand;

    public AbstractLayer(int size) {
        this.size = size;
        this.biases = new double[size];
        this.rand = new Random();
        this.randomBiases();
    }

    private void randomBiases() {
        for (int i = 0; i < this.biases.length; i++) {
            // Using bell curve to lessen chance of 0 or 1 initial bias
            this.biases[i] = 2 * this.rand.nextDouble() - 1;
        }
    }

    @Override
    public double[][] getInputWeights() {
        return this.weights;
    }

    @Override
    public void setInputWeights(double[][] weights) {
        if (weights.length != size || weights[0].length != inputSize) {
            throw new IllegalArgumentException("Invalid weight matrix");
        }

        this.weights = weights;
    }

    @Override
    public double[] getBiases() {
        return this.biases;
    }

    @Override
    public void setBiases(double[] biases) {
        if (biases.length != this.size) {
            throw new IllegalArgumentException("Invalid bias size");
        }

        this.biases = biases;
    }

    @Override
    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
    }

    @Override
    public int outputSize() {
        return this.size;
    }

    @Override
    public double[] process(double[] input) {
        double[] output = new double[this.size];

        for (int i = 0; i < this.size; i++) {
            output[i] = Util.sigmoid(Util.dotProduct(this.weights[i], input) + this.biases[i]);
        }

        this.lastOutput = output;

        return output;
    }

    private void updateWeights(double[] errors, double learningRate, double[] prevLayerOutputs) {
        for (int i = 0; i < this.size; i++) {
            for (int j = 0; j < this.inputSize; j++) {
                this.weights[i][j] += learningRate * errors[i] * prevLayerOutputs[j];
            }
        }
    }

    private void updateBiases(double[] errors, double learningRate) {
        for (int i = 0; i < this.size; i++) {
            this.biases[i] += learningRate * errors[i];
        }
    }

    @Override
    public double[] outputTrain(double[] expected, double learningRate, double[] prevLayerOutputs) {
        double[] errors = new double[expected.length];

        for (int i = 0; i < errors.length; i++) {
            errors[i] = (expected[i] - lastOutput[i]) * lastOutput[i] * (1 - lastOutput[i]);
        }

        updateWeights(errors, learningRate, prevLayerOutputs);
        updateBiases(errors, learningRate);

        return errors;
    }

    @Override
    public double[] hiddenTrain(double[] futureError, double[][] outgoingWeights, double learningRate, double[] prevLayerOutputs) {
        double[] errors = new double[this.size];

        for (int i = 0; i < errors.length; i++) {
            double sum = 0;

            for (int j = 0; j < futureError.length; j++) {
                sum += outgoingWeights[j][i] * futureError[j];
            }

            errors[i] = lastOutput[i] * (1 - lastOutput[i]) * sum;
        }

        updateWeights(errors, learningRate, prevLayerOutputs);
        updateBiases(errors, learningRate);

        return errors;
    }

    protected void createMatrix() {
        this.weights = new double[this.size][this.inputSize];
    }

    @Override
    public double[] getLastOutput() {
        return this.lastOutput;
    }

    @Override
    public String toString() {
        // Save as comma separated string
        StringBuilder res = new StringBuilder();

        for (int i = 0; i < this.weights.length; i++) {
            for (int j = 0; j < this.weights[i].length; j++) {
                res.append(this.weights[i][j] + ",");
            }
        }

        for (int i = 0; i < this.biases.length; i++) {
            res.append(this.biases[i] + ",");
        }

        return res.toString();
    }

    @Override 
    public void load(String str) {
        String[] split = str.split(",");

        int index = 0;

        for (int i = 0; i < this.weights.length; i++) {
            for (int j = 0; j < this.weights[i].length; j++) {
                this.weights[i][j] = Double.parseDouble(split[index++]);
            }
        }

        for (int i = 0; i < this.biases.length; i++) {
            this.biases[i] = Double.parseDouble(split[index++]);
        }
    }
}
