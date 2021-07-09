package mnist.network;

public class Util {

    public static double dotProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Mismatching array lengths");
        }

        double output = 0;

        for (int i = 0; i < a.length; i++) {
            output += a[i] * b[i];
        }

        return output;
    }

    public static double sigmoid(double val) {
        return 1.0 / (1.0 + Math.pow(Math.E, 0 - val));
    }
}
