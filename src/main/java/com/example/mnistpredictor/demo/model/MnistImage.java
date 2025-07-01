package com.example.mnistpredictor.demo.model;


public class MnistImage {
    private final double[] pixelData; // Flattened 784-element array for 28x28, normalized 0.0-1.0
    private final int label;           // The actual digit (0-9)

    public MnistImage(double[] pixelData, int label) {
        this.pixelData = pixelData;
        this.label = label;
    }

    public double[] getPixelData() {
        return pixelData;
    }

    public int getLabel() {
        return label;
    }
}