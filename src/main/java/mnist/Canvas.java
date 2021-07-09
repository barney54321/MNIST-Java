package mnist;

public class Canvas {
    private static int Y_OFFSET = 50;
    private static int X_OFFSET = 100;

    private double[][] grid;

    public Canvas() {
        this.grid = new double[28][28];
    }

    public void clear() {
        this.grid = new double[28][28];
    }

    public void mouseDown(int x, int y) {
        if (x > X_OFFSET && x < X_OFFSET + 28 * App.GRID_WIDTH && y > Y_OFFSET && y < Y_OFFSET + 28 * App.GRID_WIDTH) {
            int xGrid = (x - X_OFFSET) / App.GRID_WIDTH;
            int yGrid = (y - Y_OFFSET) / App.GRID_WIDTH;

            this.grid[yGrid][xGrid] = 1;
        }
    }

    public void draw(App app) {
        // Draw border
        app.fill(255, 255, 255);
        app.rect(X_OFFSET, Y_OFFSET, 28 * App.GRID_WIDTH, 28 * App.GRID_WIDTH);

        app.fill(0, 0, 0);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                if (this.grid[i][j] != 0) {
                    app.rect(X_OFFSET + j * App.GRID_WIDTH, Y_OFFSET + i * App.GRID_WIDTH, App.GRID_WIDTH, App.GRID_WIDTH);
                }
            }
        }
    }

    public double[][] getData() {
        return this.grid;
    }
}