package com.example.mnistpredictor.demo.util;

public class ImagePreProcessor {

    /**
     * Normalizes pixel values from 0-255 to 0.0-1.0 and flattens a 2D image byte array into a 1D double array.
     *
     * @param imageData A 1D byte array representing an image (e.g., 784 bytes for 28x28).
     * @return A 1D double array with normalized pixel values.
     */
    public static double[] normalizeAndFlatten(byte[] imageData) {
        double[] normalized = new double[imageData.length];
        for (int i = 0; i < imageData.length; i++) {
            // Convert byte to unsigned int (0-255) and normalize to 0.0-1.0
            normalized[i] = (imageData[i] & 0xFF) / 255.0;
        }
        return normalized;
    }
}