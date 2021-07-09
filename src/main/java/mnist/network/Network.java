package mnist.network;

import mnist.network.layer.Layer;

import java.io.*;
import java.util.*;

public class Network {

    private int inputSize;
    private int outputSize;
    private List<Layer> layers;
    private double trainingRate;

    public Network(int input, int output, List<Layer> layers, double trainingRate) {
        this.inputSize = input;
        this.outputSize = output;
        this.layers = layers;
        this.trainingRate = trainingRate;
    }

    public void train(double[] input, double[] expected) {
        if (input.length != inputSize || expected.length != outputSize) {
            throw new IllegalArgumentException("Mismatch");
        }

        run(input);

        // Get the output layer
        Layer outputLayer = this.layers.get(this.layers.size() - 1);

        double[] errors = outputLayer.outputTrain(expected, this.trainingRate, this.layers.get(this.layers.size() - 2).getLastOutput());

        for (int i = this.layers.size() - 2; i >= 0; i--) {
            Layer layer = this.layers.get(i);

            if (i != 0) {
                errors = layer.hiddenTrain(errors, this.layers.get(i + 1).getInputWeights(), this.trainingRate, this.layers.get(i - 1).getLastOutput());
            } else {
                errors = layer.hiddenTrain(errors, this.layers.get(i + 1).getInputWeights(), this.trainingRate, input);
            }
        }
    }

    public boolean compareExpectedWithOutput(int expected, double[] output) {
        int oMax = 0;

        for (int i = 0; i < outputSize; i++) {
            if (output[i] > output[oMax])
                oMax = i;
        }

        return expected == oMax;
    }

    public double[] run(double[] input) {
        double[] prevOutput = input;

        for (Layer layer : this.layers) {
            prevOutput = layer.process(prevOutput);
        }

        return prevOutput;
    }

    public List<Layer> getLayers() {
        return this.layers;
    }

    public void setTrainingRate(double rate) {
        this.trainingRate = rate;
    }

    public void save(String path) {
        // Each line represents a layer
        File f = new File(path);

        try {
            PrintWriter writer = new PrintWriter(f);

            int count = 0;

            for (Layer layer : this.layers) {
                System.out.println("Saving layer: " + (++count));
                writer.println(layer.toString());
            }

            writer.flush();
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException("Failed to save");
        }
    }

    public void load(String path) {
        File f = new File(path);

        try {
            Scanner sc = new Scanner(f);

            int count = 0;
            for (Layer layer : this.layers) {
                System.out.println("Loading layer: " + (++count));
                layer.load(sc.nextLine());
            }

            sc.close();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load");
        }
    }
}
