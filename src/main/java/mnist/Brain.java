package mnist;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import mnist.network.*;
import mnist.network.layer.*;

public class Brain {

    private Canvas canvas;

    private Network network;

    private int prediction;

    private class Example {
        private int label;

        private double[] data;

        public Example(String line) {
            String[] split = line.split(",");
            this.label = Integer.parseInt(split[0]);
            this.data = new double[28 * 28];

            for (int i = 1; i < split.length; i++) {
                this.data[i - 1] = Double.parseDouble(split[i]) / 255.0;
            }
        }

        public int getLabel() {
            return this.label;
        }

        public double[] getData() {
            return this.data;
        }
    }

    public Brain(Canvas canvas) {
        this.canvas = canvas;

        this.prediction = 0;

        this.network = new NetworkBuilder().setInput(28 * 28)
                                           .setOutput(10)
                                           .appendHiddenLayer(new FullLayer(400))
                                           .appendHiddenLayer(new FullLayer(205))
                                           .setTrainingRate(0.1)
                                           .build();
    }

    public void predict() {
        double[] input = flatten(this.canvas.getData());
        
        double[] output = this.network.run(input);

        this.prediction = getMaxIndex(output);
    }

    private double[] flatten(double[][] arr) {
        double[] res = new double[arr.length * arr[0].length];

        int index = 0;

        for (double[] sub : arr) {
            for (double d : sub) {
                res[index++] = d;
            }
        }

        return res;
    }

    private int getMaxIndex(double[] arr) {
        int max = 0;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > arr[max]) {
                max = i;
            }
        }

        return max;
    }

    public void draw(App app) {
        app.textSize(40);
        app.text(this.prediction + "", 100 + App.GRID_WIDTH * 14 - 20, 400);
    }

    public void train() {

        List<Example> tests = new ArrayList<>();
        List<Example> examples = new ArrayList<>();

        File csv = new File("src/main/resources/mnist_train.csv");

        try {
            Scanner sc = new Scanner(csv);

            // Ignore header line
            sc.nextLine();

            while (sc.hasNextLine()) {
                examples.add(new Example(sc.nextLine()));
            }

            sc.close();

        } catch (IOException e) {
            throw new Error("Fail");
        }

        csv = new File("src/main/resources/mnist_test.csv");

        try {
            Scanner sc = new Scanner(csv);

            // Ignore header line
            sc.nextLine();

            while (sc.hasNextLine()) {
                tests.add(new Example(sc.nextLine()));
            }

            sc.close();

        } catch (IOException e) {
            throw new Error("Fail");
        }

        for (int i = 0; i < 10; i++) {
            System.out.println("EPOCH: " + i);

            Collections.shuffle(examples);

            for (int j = 0; j < examples.size(); j++) {
                Example example = examples.get(j);

                double[] expected = new double[10];
                expected[example.getLabel()] = 1;

                this.network.train(example.getData(), expected);

                double percentage = ((double) j / (double) examples.size()) * 100;
                System.out.print((int) percentage + "%\t\r");
            }

            System.out.print("\r100%\t\t\t\n");

            int correct = 0;

            for (int j = 0; j < tests.size(); j++) {
                Example example = tests.get(j);

                double[] output = network.run(example.getData());

                if (this.network.compareExpectedWithOutput(example.getLabel(), output)) {
                    correct++;
                }
            }

            System.out.println("Correct: " + correct + "/" + tests.size());
        }

        System.out.println("DONE");
    }

    public void save(String path) {
        // Save the network to the file
        System.out.println("Saving to file");
        this.network.save(path);
    }

    public void load(String path) {
        // Load the network from the file
        this.network.load(path);
    }
}