package mnist.network;

import mnist.network.layer.FullLayer;
import mnist.network.layer.Layer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    private int input;
    private int output;
    private double trainingRate = 0.1;

    private List<Layer> layers;

    public NetworkBuilder() {
        this.layers = new ArrayList<>();
    }

    public NetworkBuilder setInput(int size) {
        this.input = size;
        return this;
    }

    public NetworkBuilder setOutput(int size) {
        this.output = size;
        return this;
    }

    public NetworkBuilder appendHiddenLayer(Layer layer) {
        this.layers.add(layer);
        return this;
    }

    public NetworkBuilder prependHiddenLayer(Layer layer) {
        this.layers.add(0, layer);
        return this;
    }

    public NetworkBuilder insertHiddenLayer(Layer layer, int index) {
        this.layers.add(index, layer);
        return this;
    }

    public NetworkBuilder setTrainingRate(double rate) {
        this.trainingRate = rate;
        return this;
    }

    public Network build() {
        if (this.input <= 0 || this.output <= 0) {
            throw new IllegalStateException("Invalid input/output size(s).");
        }

        this.layers.add(new FullLayer(output));

        this.layers.get(0).setInputSize(input);

        for (int i = 1; i < layers.size(); i++) {
            this.layers.get(i).setInputSize(this.layers.get(i - 1).outputSize());
        }

        for (Layer layer : this.layers) {
            layer.setUpWeights();
        }

        return new Network(input, output, layers, trainingRate);
    }

}
