package com.example.mnistpredictor.demo.ml.randomforest;

import com.example.mnistpredictor.demo.model.MnistImage;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.io.IOException;


public class DecisionTree implements Serializable {
    private static final long serialVersionUID = 1L;

    private Node root;
    private int maxDepth;          // Maximum depth of the tree
    private int minSamplesSplit;   // Minimum number of samples required to split an internal node
    private int numFeatures;       // Total number of features in the dataset (e.g., 784 for MNIST)
    private int numRandomFeatures; // Number of random features to consider at each split (for Random Forest)
    private transient Random random; // Transient for serialization reasons

    public DecisionTree(int maxDepth, int minSamplesSplit, int numFeatures, int numRandomFeatures, Random random) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.numFeatures = numFeatures;
        this.numRandomFeatures = numRandomFeatures;
        this.random = random;
    }

    // Custom deserialization to re-initialize the transient Random object
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        this.random = new Random();
    }

    // Node class representing a node in the decision tree
    private static class Node implements Serializable {
        private static final long serialVersionUID = 1L;
        int featureIndex;     // Index of the feature used for splitting at this node
        double threshold;      // Threshold value for the split (feature_value <= threshold goes left)
        int predictedClass;   // Predicted class if this is a leaf node
        Node left;            // Left child node
        Node right;           // Right child node
        boolean isLeaf;       // True if this node is a leaf

        // Constructor for a leaf node
        Node(int predictedClass) {
            this.predictedClass = predictedClass;
            this.isLeaf = true;
        }

        // Constructor for an internal node
        Node(int featureIndex, double threshold) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.isLeaf = false;
        }
    }

    /**
     * Trains the decision tree using the provided training data.
     * @param trainingData A subset of data (e.g., a bootstrapped sample for Random Forest).
     */
    public void train(List<MnistImage> trainingData) {
        this.root = buildTree(trainingData, 0);
    }

    /**
     * Recursively builds the decision tree.
     * @param currentData The subset of data for the current node.
     * @param depth The current depth of the tree.
     * @return The constructed Node.
     */
    private Node buildTree(List<MnistImage> currentData, int depth) {
        if (currentData.isEmpty()) {
            return null;
        }

        // Stopping conditions:
        // 1. All samples in currentData have the same label (pure node)
        int firstLabel = currentData.get(0).getLabel();
        boolean allSameLabel = true;
        for (MnistImage img : currentData) {
            if (img.getLabel() != firstLabel) {
                allSameLabel = false;
                break;
            }
        }
        if (allSameLabel) {
            return new Node(firstLabel);
        }

        // 2. Max depth reached
        if (depth >= maxDepth) {
            return new Node(getMajorityClass(currentData));
        }

        // 3. Not enough samples to split
        if (currentData.size() < minSamplesSplit) {
            return new Node(getMajorityClass(currentData));
        }

        // Find the best split (feature and threshold)
        int bestFeature = -1;
        double bestThreshold = -1.0;
        double bestGini = Double.MAX_VALUE;

        // Randomly select a subset of features to consider for splitting (Random Forest characteristic)
        List<Integer> candidateFeatures = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            candidateFeatures.add(i);
        }
        Collections.shuffle(candidateFeatures, random); // Shuffle to pick random features
        List<Integer> featuresToConsider = candidateFeatures.subList(0, Math.min(numRandomFeatures, numFeatures));


        for (int featureIndex : featuresToConsider) {
            // Collect unique values for threshold candidates for this feature
            // Any value between two distinct feature values can be a threshold.
            // For simplicity, we just use existing feature values as thresholds.
            Set<Double> thresholds = new HashSet<>();
            for (MnistImage img : currentData) {
                thresholds.add(img.getPixelData()[featureIndex]);
            }

            for (double threshold : thresholds) {
                List<MnistImage> leftSplit = new ArrayList<>();  // Samples with feature_value <= threshold
                List<MnistImage> rightSplit = new ArrayList<>(); // Samples with feature_value > threshold

                for (MnistImage img : currentData) {
                    if (img.getPixelData()[featureIndex] <= threshold) {
                        leftSplit.add(img);
                    } else {
                        rightSplit.add(img);
                    }
                }

                // A split is only useful if it divides the data into two non-empty sets
                if (leftSplit.isEmpty() || rightSplit.isEmpty()) {
                    continue;
                }

                // Calculate Gini impurity for this split
                double gini = calculateGiniImpurity(leftSplit, rightSplit);

                // Update best split if current split is better
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = featureIndex;
                    bestThreshold = threshold;
                }
            }
        }

        // If no valid split was found (e.g., all samples have identical feature values),
        // make this node a leaf node predicting the majority class.
        if (bestFeature == -1) {
            return new Node(getMajorityClass(currentData));
        }

        // Create child nodes based on the best split
        List<MnistImage> leftData = new ArrayList<>();
        List<MnistImage> rightData = new ArrayList<>();
        for (MnistImage img : currentData) {
            if (img.getPixelData()[bestFeature] <= bestThreshold) {
                leftData.add(img);
            } else {
                rightData.add(img);
            }
        }

        Node node = new Node(bestFeature, bestThreshold);
        node.left = buildTree(leftData, depth + 1);  // Recursively build left child
        node.right = buildTree(rightData, depth + 1); // Recursively build right child
        return node;
    }

    /**
     * Calculates the weighted Gini impurity for a split.
     * Gini_split = (left_size/total_size) * Gini(left_node) + (right_size/total_size) * Gini(right_node)
     */
    private double calculateGiniImpurity(List<MnistImage> left, List<MnistImage> right) {
        int totalSize = left.size() + right.size();
        if (totalSize == 0) return 0.0;

        double giniLeft = calculateNodeGini(left);
        double giniRight = calculateNodeGini(right);

        return ((double) left.size() / totalSize * giniLeft) + ((double) right.size() / totalSize * giniRight);
    }

    /**
     * Calculates Gini impurity for a single node (subset of data).
     * Gini = 1 - sum( (proportion_of_class_k)^2 ) for all classes k
     */
    private double calculateNodeGini(List<MnistImage> data) {
        if (data.isEmpty()) return 0.0;

        // Count occurrences of each class label in the data subset
        Map<Integer, Long> classCounts = data.stream()
                .collect(Collectors.groupingBy(MnistImage::getLabel, Collectors.counting()));

        double gini = 1.0;
        for (Long count : classCounts.values()) {
            double p = (double) count / data.size(); // Proportion of this class
            gini -= (p * p); // Subtract squared proportion
        }
        return gini;
    }

    /**
     * Returns the majority class label in a given dataset. Used for leaf node prediction.
     */
    private int getMajorityClass(List<MnistImage> data) {
        if (data.isEmpty()) {
            return -1; // Should not happen if buildTree handles empty data correctly
        }
        Map<Integer, Long> classCounts = data.stream()
                .collect(Collectors.groupingBy(MnistImage::getLabel, Collectors.counting()));

        // Find the class with the maximum count
        return classCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(-1); // Fallback, should not be reached if data is not empty
    }

    /**
     * Predicts the class for a single feature vector by traversing the trained tree.
     * @param features Input feature vector.
     * @return Predicted class label.
     */
    public int predict(double[] features) {
        if (root == null) {
            throw new IllegalStateException("Decision tree not trained.");
        }
        return traverseTree(features, root);
    }

    /**
     * Recursive helper method to traverse the tree for prediction.
     */
    private int traverseTree(double[] features, Node node) {
        if (node.isLeaf) {
            return node.predictedClass; // If it's a leaf, return its predicted class
        }

        // If it's an internal node, decide which child to go to based on the feature and threshold
        if (features[node.featureIndex] <= node.threshold) {
            return traverseTree(features, node.left);
        } else {
            return traverseTree(features, node.right);
        }
    }
}