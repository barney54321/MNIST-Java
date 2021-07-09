package mnist.network.layer;

public interface Layer {

    double[][] getInputWeights();

    void setInputWeights(double[][] weights);

    double[] getBiases();

    void setBiases(double[] biases);

    int outputSize();

    void setInputSize(int size);

    void setUpWeights();

    double[] process(double[] input);

    double[] outputTrain(double[] expected, double learningRate, double[] prevLastOutput);

    double[] hiddenTrain(double[] futureError, double[][] outgoingWeights, double learningRate, double[] prevLastOutput);

    double[] getLastOutput();

    String toString();

    void load(String str);
}
