package mnist;

import processing.core.PApplet;

public class App extends PApplet {

    public static final int GRID_WIDTH = 10;

    public static final int HEIGHT = 500;
    public static final int WIDTH = 200 + GRID_WIDTH * 28;

    private Canvas canvas;

    private Brain brain;

    private boolean mouseDown;

    public App() {
        this.canvas = new Canvas();
        this.brain = new Brain(this.canvas);
        this.brain.load("network");
        this.mouseDown = false;
    }

    public void setup() {
        frameRate(60);
    }

    public void settings() {
        size(WIDTH, HEIGHT);
    }

    public void draw() {
        background(150, 150, 150);
        this.canvas.draw(this);
        this.brain.draw(this);

        if (this.mouseDown) {
            this.canvas.mouseDown(pmouseX, pmouseY);
        }

        this.brain.predict();
    }

    public void mousePressed() {
        this.mouseDown = true;
    }

    public void mouseReleased() {
        this.mouseDown = false;
    }

    public void keyPressed() {
        if (keyCode == 32) {
            this.canvas.clear();
        }
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            PApplet.main("mnist.App");
        } else {
            // If given arguments, run training
            Brain brain = new Brain(null);
            brain.train();
            brain.save("network");
        }
    }

}
