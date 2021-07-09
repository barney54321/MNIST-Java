package mnist.network.layer;

public class FullLayer extends AbstractLayer {

    public FullLayer(int size) {
        super(size);
    }

    @Override
    public void setUpWeights() {
        this.createMatrix();

        for (int i = 0; i < this.weights.length; i++) {
            for (int j = 0; j < this.weights[i].length; j++) {
                this.weights[i][j] = 2 * rand.nextDouble() - 1;
            }
        }
    }
}
