package com.example.mnistpredictor.demo.ml.logisticregression;

import com.example.mnistpredictor.demo.ml.Classifier;
import com.example.mnistpredictor.demo.model.MnistImage;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class LogisticRegressionClassifier implements Classifier, Serializable{
    private static final long serialVersionUID = 1L;

    private Map<Integer, BinaryLogisticRegression> binaryClassifiers;
    private int numFeatures;
    private double learningRate;
    private int numEpochs;
    private int numClasses;

    public LogisticRegressionClassifier(int numFeatures, int numClasses, double learningRate, int numEpochs){
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.learningRate = learningRate;
        this.numEpochs = numEpochs;
        this.binaryClassifiers = new HashMap<>();
    }

    @Override
    public void train(List<MnistImage> trainingData) throws Exception{
        System.out.println("Training LR (OvR) classifier");
        for (int k = 0; k < numClasses; k++){
            System.out.println("Training for digit " + k);
            BinaryLogisticRegression binaryClassifier = new BinaryLogisticRegression(numFeatures);
            // features
            List<double[]> X_binary = new ArrayList<>();
            // labels (1 if digit 'k', 0 otherwise)
            List<Integer> y_binary = new ArrayList<>();

            for (MnistImage image : trainingData) {
                X_binary.add(image.getPixelData());
                // For the current digit 'k', if the image's true label is 'k', assign 1; otherwise assign 0.
                y_binary.add(image.getLabel() == k ? 1 : 0);
            }

            // Train once per class using the fully populated dataset
            binaryClassifier.train(X_binary, y_binary, learningRate, numEpochs);

            // Store the trained binary classifier for digit `k`
            binaryClassifiers.put(k, binaryClassifier);
            
            System.out.println("Logistic Regression training complete.");
        }
    }
    
    @Override
    public int predict(double[] features) throws Exception{
        if (features.length != numFeatures){
            throw new IllegalArgumentException("Input features dim mismatch. Expected " + numFeatures + "got " + features.length);
        }
        if (binaryClassifiers.isEmpty()) throw new IllegalArgumentException("LR model was not trained");
        
        double maxProbability = Double.NEGATIVE_INFINITY;
        int predictedClass = -1;
        
        // For each binary classifier, compute probability and keep the highest
        for (Map.Entry<Integer, BinaryLogisticRegression> entry : binaryClassifiers.entrySet()) {
            int digit = entry.getKey();
            BinaryLogisticRegression classifier = entry.getValue();

            double probability = classifier.predictProbability(features);
            if (probability > maxProbability) {
                maxProbability = probability;
                predictedClass = digit;
            }
        }
        return predictedClass;
    }
}

