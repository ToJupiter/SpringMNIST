package com.example.mnistpredictor.demo;

import com.example.mnistpredictor.demo.ml.logisticregression.LogisticRegressionClassifier;
import com.example.mnistpredictor.demo.ml.randomforest.RandomForestClassifier;
import com.example.mnistpredictor.demo.ml.kmeans.KMeansClusterer;
import com.example.mnistpredictor.demo.ml.Classifier;
import com.example.mnistpredictor.demo.ml.Clusterer;
import com.example.mnistpredictor.demo.model.MnistImage;
import com.example.mnistpredictor.demo.util.ImagePreProcessor;
import com.example.mnistpredictor.demo.util.MnistDataReader;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;

public class ModelStatsRunner {
    public static void main(String[] args) throws Exception {
        // Create results directory
        File resultsDir = new File("results");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        
        // Load MNIST data
        System.out.println("Loading MNIST dataset...");
        String dataDir = "src/main/resources/data/";
        List<MnistImage> train = loadMnist(dataDir + "train-images-idx3-ubyte", dataDir + "train-labels-idx1-ubyte");
        List<MnistImage> test = loadMnist(dataDir + "t10k-images-idx3-ubyte", dataDir + "t10k-labels-idx1-ubyte");
        
        System.out.println("Training set size: " + train.size());
        System.out.println("Test set size: " + test.size());
        System.out.println();

        // Train and evaluate each model
        System.out.println("Training and evaluating Logistic Regression...");
        LogisticRegressionClassifier lr = new LogisticRegressionClassifier(784, 10, 0.01, 200);
        long startTime = System.currentTimeMillis();
        lr.train(train);
        long trainingTime = System.currentTimeMillis() - startTime;
        writeEnhancedStats(lr, test, "results/logistic_regression_results.txt", "Logistic Regression", trainingTime);

        System.out.println("Training and evaluating Random Forest...");
        RandomForestClassifier rf = new RandomForestClassifier(20, 20, 5, 784, (int)Math.sqrt(784));
        startTime = System.currentTimeMillis();
        rf.train(train);
        trainingTime = System.currentTimeMillis() - startTime;
        writeEnhancedStats(rf, test, "results/random_forest_results.txt", "Random Forest", trainingTime);

        System.out.println("Training and evaluating K-Means Clustering...");
        KMeansClusterer km = new KMeansClusterer(10, 784, 100);
        startTime = System.currentTimeMillis();
        km.train(train);
        trainingTime = System.currentTimeMillis() - startTime;
        writeKMeansStats(km, test, "results/kmeans_results.txt", trainingTime);

        // Write summary comparison
        writeSummaryComparison("results/model_comparison_summary.txt");
        
        System.out.println("All model evaluation reports have been generated in the 'results/' directory:");
        System.out.println("  - logistic_regression_results.txt");
        System.out.println("  - random_forest_results.txt");
        System.out.println("  - kmeans_results.txt");
        System.out.println("  - model_comparison_summary.txt");
    }

    private static List<MnistImage> loadMnist(String imagesPath, String labelsPath) throws IOException {
        try (InputStream imgStream = new FileInputStream(imagesPath);
             InputStream lblStream = new FileInputStream(labelsPath)) {
            byte[][] images = MnistDataReader.readImages(imgStream);
            byte[] labels = MnistDataReader.readLabels(lblStream);
            List<MnistImage> data = new ArrayList<>(images.length);
            for (int i = 0; i < images.length; i++) {
                data.add(new MnistImage(ImagePreProcessor.normalizeAndFlatten(images[i]), labels[i]));
            }
            return data;
        }
    }

    private static void writeEnhancedStats(Classifier model, List<MnistImage> test, String outPath, String modelName, long trainingTimeMs) throws Exception {
        // Create results directory if it doesn't exist
        File parentDir = new File(outPath).getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        int[][] confusion = new int[10][10];
        int correct = 0;
        int[] totalPerClass = new int[10];
        int[] correctPerClass = new int[10];
        
        // Make predictions and collect statistics
        for (MnistImage img : test) {
            int actual = img.getLabel();
            int predicted = model.predict(img.getPixelData());
            if (actual >= 0 && actual < 10 && predicted >= 0 && predicted < 10) {
                confusion[actual][predicted]++;
            }
            if (actual == predicted) {
                correct++;
                correctPerClass[actual]++;
            }
            totalPerClass[actual]++;
        }
        
        double accuracy = (double) correct / test.size();
        
        try (PrintWriter out = new PrintWriter(outPath)) {
            // Header information
            out.println("=".repeat(80));
            out.println(modelName.toUpperCase() + " - EVALUATION REPORT");
            out.println("=".repeat(80));
            out.println("Generated on: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            out.println("Training time: " + String.format("%.2f", trainingTimeMs / 1000.0) + " seconds");
            out.println("Test samples: " + test.size());
            out.println();
            
            // Overall accuracy
            out.println("OVERALL PERFORMANCE");
            out.println("-".repeat(40));
            out.printf("Overall Accuracy: %.4f (%d/%d correct predictions)\n", accuracy, correct, test.size());
            out.printf("Overall Error Rate: %.4f\n", 1.0 - accuracy);
            out.println();
            
            // Confusion Matrix
            out.println("CONFUSION MATRIX");
            out.println("-".repeat(40));
            out.println("Rows represent actual classes, columns represent predicted classes");
            out.print("     ");
            for (int j = 0; j < 10; j++) {
                out.printf("%6d", j);
            }
            out.println();
            
            for (int i = 0; i < 10; i++) {
                out.printf("%3d: ", i);
                for (int j = 0; j < 10; j++) {
                    out.printf("%6d", confusion[i][j]);
                }
                out.println();
            }
            out.println();
            
            // Per-class metrics
            out.println("CLASSIFICATION REPORT (similar to sklearn)");
            out.println("-".repeat(80));
            out.printf("%-5s %9s %9s %9s %9s\n", "Class", "Precision", "Recall", "F1-Score", "Support");
            out.println("-".repeat(80));
            
            double[] precision = new double[10];
            double[] recall = new double[10];
            double[] f1Score = new double[10];
            double macroAvgPrecision = 0, macroAvgRecall = 0, macroAvgF1 = 0;
            double weightedAvgPrecision = 0, weightedAvgRecall = 0, weightedAvgF1 = 0;
            
            for (int i = 0; i < 10; i++) {
                int tp = confusion[i][i];
                int fp = 0, fn = 0;
                
                // Calculate false positives and false negatives
                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        fp += confusion[j][i];  // false positives for class i
                        fn += confusion[i][j];  // false negatives for class i
                    }
                }
                
                precision[i] = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
                recall[i] = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
                f1Score[i] = precision[i] + recall[i] == 0 ? 0 : 2 * precision[i] * recall[i] / (precision[i] + recall[i]);
                
                out.printf("%-5d %9.4f %9.4f %9.4f %9d\n", i, precision[i], recall[i], f1Score[i], totalPerClass[i]);
                
                // Accumulate for macro averages
                macroAvgPrecision += precision[i];
                macroAvgRecall += recall[i];
                macroAvgF1 += f1Score[i];
                
                // Accumulate for weighted averages
                weightedAvgPrecision += precision[i] * totalPerClass[i];
                weightedAvgRecall += recall[i] * totalPerClass[i];
                weightedAvgF1 += f1Score[i] * totalPerClass[i];
            }
            
            // Calculate averages
            macroAvgPrecision /= 10;
            macroAvgRecall /= 10;
            macroAvgF1 /= 10;
            weightedAvgPrecision /= test.size();
            weightedAvgRecall /= test.size();
            weightedAvgF1 /= test.size();
            
            out.println("-".repeat(80));
            out.printf("%-5s %9.4f %9.4f %9.4f %9d\n", "macro", macroAvgPrecision, macroAvgRecall, macroAvgF1, test.size());
            out.printf("%-5s %9.4f %9.4f %9.4f %9d\n", "wtd", weightedAvgPrecision, weightedAvgRecall, weightedAvgF1, test.size());
            out.println("-".repeat(80));
            out.println();
            
            // Additional statistics
            out.println("ADDITIONAL STATISTICS");
            out.println("-".repeat(40));
            
            // Per-class accuracy
            out.println("Per-class Accuracy:");
            for (int i = 0; i < 10; i++) {
                double classAccuracy = totalPerClass[i] == 0 ? 0 : (double) correctPerClass[i] / totalPerClass[i];
                out.printf("  Class %d: %.4f (%d/%d)\n", i, classAccuracy, correctPerClass[i], totalPerClass[i]);
            }
            out.println();
            
            // Top misclassifications
            out.println("Top Misclassifications:");
            List<Map.Entry<String, Integer>> misclassifications = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    if (i != j && confusion[i][j] > 0) {
                        misclassifications.add(new AbstractMap.SimpleEntry<>(
                            "Class " + i + " misclassified as " + j, confusion[i][j]));
                    }
                }
            }
            
            misclassifications.sort((a, b) -> b.getValue().compareTo(a.getValue()));
            for (int i = 0; i < Math.min(10, misclassifications.size()); i++) {
                Map.Entry<String, Integer> entry = misclassifications.get(i);
                out.printf("  %s: %d times\n", entry.getKey(), entry.getValue());
            }
            out.println();
            
            out.println("=".repeat(80));
            out.println("End of " + modelName + " evaluation report");
            out.println("=".repeat(80));
        }
    }

    private static void writeKMeansStats(KMeansClusterer km, List<MnistImage> test, String outPath, long trainingTimeMs) throws Exception {
        // Create results directory if it doesn't exist
        File parentDir = new File(outPath).getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        Map<Integer, List<Integer>> clusterToLabels = new HashMap<>();
        Map<Integer, Map<Integer, Integer>> clusterLabelCounts = new HashMap<>();
        
        for (MnistImage img : test) {
            int cluster = km.cluster(img.getPixelData());
            clusterToLabels.computeIfAbsent(cluster, k -> new ArrayList<>()).add(img.getLabel());
            clusterLabelCounts.computeIfAbsent(cluster, k -> new HashMap<>())
                             .merge(img.getLabel(), 1, Integer::sum);
        }
        
        try (PrintWriter out = new PrintWriter(outPath)) {
            out.println("=".repeat(80));
            out.println("K-MEANS CLUSTERING - EVALUATION REPORT");
            out.println("=".repeat(80));
            out.println("Generated on: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            out.println("Training time: " + String.format("%.2f", trainingTimeMs / 1000.0) + " seconds");
            out.println("Test samples: " + test.size());
            out.println("Number of clusters (K): 10");
            out.println();
            
            out.println("CLUSTER ANALYSIS");
            out.println("-".repeat(80));
            out.printf("%-7s %-10s %-15s %-40s\n", "Cluster", "Size", "Dominant Label", "Label Distribution");
            out.println("-".repeat(80));
            
            double totalPurity = 0;
            int[] clusterSizes = new int[10];
            
            for (int cluster = 0; cluster < 10; cluster++) {
                List<Integer> labels = clusterToLabels.getOrDefault(cluster, Collections.emptyList());
                clusterSizes[cluster] = labels.size();
                
                if (labels.isEmpty()) {
                    out.printf("%-7d %-10d %-15s %-40s\n", cluster, 0, "None", "Empty cluster");
                    continue;
                }
                
                // Find dominant label and calculate purity
                Map<Integer, Integer> labelCounts = clusterLabelCounts.get(cluster);
                int dominantLabel = labelCounts.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(-1);
                int dominantCount = labelCounts.getOrDefault(dominantLabel, 0);
                double purity = (double) dominantCount / labels.size();
                totalPurity += purity * labels.size();
                
                // Create distribution string
                StringBuilder distribution = new StringBuilder();
                for (int i = 0; i < 10; i++) {
                    int count = labelCounts.getOrDefault(i, 0);
                    if (count > 0) {
                        if (distribution.length() > 0) distribution.append(" ");
                        distribution.append(String.format("%d:%d", i, count));
                    }
                }
                
                out.printf("%-7d %-10d %-15s %-40s\n", 
                    cluster, labels.size(), 
                    dominantLabel + " (" + String.format("%.2f", purity) + ")", 
                    distribution.toString());
            }
            
            double overallPurity = totalPurity / test.size();
            out.println("-".repeat(80));
            out.printf("Overall Cluster Purity: %.4f\n", overallPurity);
            out.println();
            
            // Cluster quality metrics
            out.println("CLUSTER QUALITY METRICS");
            out.println("-".repeat(40));
            
            // Calculate silhouette-like metric (simplified version)
            out.println("Cluster size distribution:");
            for (int i = 0; i < 10; i++) {
                double percentage = (double) clusterSizes[i] / test.size() * 100;
                out.printf("  Cluster %d: %d samples (%.1f%%)\n", i, clusterSizes[i], percentage);
            }
            out.println();
            
            // Label assignment analysis
            out.println("LABEL-TO-CLUSTER ASSIGNMENT ANALYSIS");
            out.println("-".repeat(40));
            out.printf("%-5s %-15s %-20s\n", "Label", "Primary Cluster", "Assignment Distribution");
            out.println("-".repeat(40));
            
            Map<Integer, Map<Integer, Integer>> labelToClusterCounts = new HashMap<>();
            for (int cluster = 0; cluster < 10; cluster++) {
                Map<Integer, Integer> labelCounts = clusterLabelCounts.getOrDefault(cluster, new HashMap<>());
                for (Map.Entry<Integer, Integer> entry : labelCounts.entrySet()) {
                    labelToClusterCounts.computeIfAbsent(entry.getKey(), k -> new HashMap<>())
                                      .put(cluster, entry.getValue());
                }
            }
            
            for (int label = 0; label < 10; label++) {
                Map<Integer, Integer> clusterCounts = labelToClusterCounts.getOrDefault(label, new HashMap<>());
                if (clusterCounts.isEmpty()) {
                    out.printf("%-5d %-15s %-20s\n", label, "None", "No samples");
                    continue;
                }
                
                int primaryCluster = clusterCounts.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(-1);
                int primaryCount = clusterCounts.getOrDefault(primaryCluster, 0);
                int totalForLabel = clusterCounts.values().stream().mapToInt(Integer::intValue).sum();
                
                StringBuilder distribution = new StringBuilder();
                clusterCounts.entrySet().stream()
                    .sorted(Map.Entry.<Integer, Integer>comparingByValue().reversed())
                    .limit(3)  // Show top 3 clusters
                    .forEach(entry -> {
                        if (distribution.length() > 0) distribution.append(" ");
                        distribution.append(String.format("C%d:%d", entry.getKey(), entry.getValue()));
                    });
                
                out.printf("%-5d %-15s %-20s\n", 
                    label, 
                    "C" + primaryCluster + " (" + String.format("%.1f", (double)primaryCount/totalForLabel*100) + "%)",
                    distribution.toString());
            }
            
            out.println();
            out.println("=".repeat(80));
            out.println("End of K-Means clustering evaluation report");
            out.println("=".repeat(80));
        }
    }
    
    private static void writeSummaryComparison(String outPath) throws Exception {
        try (PrintWriter out = new PrintWriter(outPath)) {
            out.println("=".repeat(80));
            out.println("MODEL COMPARISON SUMMARY");
            out.println("=".repeat(80));
            out.println("Generated on: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            out.println();
            
            out.println("This summary provides a high-level comparison of all trained models.");
            out.println("For detailed metrics, please refer to individual model reports:");
            out.println("  - logistic_regression_results.txt");
            out.println("  - random_forest_results.txt");
            out.println("  - kmeans_results.txt");
            out.println();
            
            out.println("MODEL CHARACTERISTICS:");
            out.println("-".repeat(40));
            out.println("Logistic Regression:");
            out.println("  - Type: Supervised classification");
            out.println("  - Algorithm: One-vs-Rest logistic regression");
            out.println("  - Features: Linear decision boundaries");
            out.println("  - Interpretability: High");
            out.println();
            
            out.println("Random Forest:");
            out.println("  - Type: Supervised classification");
            out.println("  - Algorithm: Ensemble of decision trees");
            out.println("  - Features: Non-linear decision boundaries");
            out.println("  - Interpretability: Medium");
            out.println();
            
            out.println("K-Means Clustering:");
            out.println("  - Type: Unsupervised clustering");
            out.println("  - Algorithm: Centroid-based clustering");
            out.println("  - Features: Discovers hidden patterns");
            out.println("  - Interpretability: Medium");
            out.println();
            
            out.println("USAGE RECOMMENDATIONS:");
            out.println("-".repeat(40));
            out.println("- Use Logistic Regression for baseline performance and interpretability");
            out.println("- Use Random Forest for better accuracy with non-linear patterns");
            out.println("- Use K-Means for exploratory data analysis and pattern discovery");
            out.println();
            
            out.println("=".repeat(80));
            out.println("End of model comparison summary");
            out.println("=".repeat(80));
        }
    }
}
