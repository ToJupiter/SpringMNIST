package com.example.mnistpredictor.demo.ml.logisticregression;

import java.io.Serializable;
import java.util.List;

public class BinaryLogisticRegression implements Serializable{
    private static final long serialVersionUID = 1L;

    private double[] weights;
    private double bias;
    private int numFeatures;

    public BinaryLogisticRegression(int numFeatures){
        this.numFeatures = numFeatures;
        this.weights = new double[numFeatures];
        this.bias = 0.0;
    }

    private double sigmoid(double z){ return 1.0/ (1.0 + Math.exp(-z));}

    private double calculateLinearOutput(double[] features){
        double z = bias;
        for (int i = 0; i < numFeatures; i++){
            z += weights[i] * features[i];
        }
        return z;
    }

    public void train(List<double[]> X, List<Integer> y, double learningRate, int numEpochs){
        if (X.isEmpty() || X.get(0).length != numFeatures) {
            throw new IllegalArgumentException("Invalid training data dimensions or empty data.");
        }

        for (int epoch = 0; epoch < numEpochs; epoch++){
            double[] dw = new double[numFeatures];
            double db = 0.0; // bias grad accumulator

            for (int i = 0; i < X.size(); i++){
                double[] features = X.get(i);
                int trueLabel = y.get(i);

                double linearOutput = calculateLinearOutput(features);
                double prediction = sigmoid(linearOutput);

                double error = prediction - trueLabel;
                
                // gradient accumulation
                for (int j = 0; j < numFeatures; j++) {
                    dw[j] += error * features[j];
                }
                db += error;
            }
            
            // update w and b using the avg grad over all samples in the epoch
            for (int j = 0; j < numFeatures; j++){
                weights[j] -= learningRate * dw[j] / X.size();
            }
            bias -= learningRate * db / X.size();
            
        }
    }


    public double predictProbability(double[] features){
        if (features.length != numFeatures) throw new IllegalArgumentException("Input dim mismatch");
        return sigmoid(calculateLinearOutput(features));
    }
}
