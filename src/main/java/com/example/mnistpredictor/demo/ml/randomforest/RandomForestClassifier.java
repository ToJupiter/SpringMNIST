package com.example.mnistpredictor.demo.ml.randomforest;

import com.example.mnistpredictor.demo.ml.Classifier;
import com.example.mnistpredictor.demo.model.MnistImage;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

public class RandomForestClassifier implements Classifier, Serializable {
    private static final long serialVersionUID = 1L;

    private List<DecisionTree> trees; // The ensemble of decision trees
    private int numTrees;             // Number of trees in the forest
    private int maxDepth;             // Max depth for each individual tree
    private int minSamplesSplit;      // Min samples to split for each individual tree
    private int numFeatures;          // Total number of features
    private int numRandomFeatures;    // Number of random features to consider at each split
    private transient Random random;  // Transient for serialization reasons

    public RandomForestClassifier(int numTrees, int maxDepth, int minSamplesSplit, int numFeatures, int numRandomFeatures) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.numFeatures = numFeatures;
        this.numRandomFeatures = numRandomFeatures;
        this.trees = new ArrayList<>(numTrees);
        this.random = new Random();
    }

    // Custom deserialization to re-initialize the transient Random object
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        this.random = new Random();
    }

    @Override
    public void train(List<MnistImage> trainingData) throws Exception {
        System.out.println("Training Random Forest classifier with " + numTrees + " trees...");

        // Use an ExecutorService to train trees in parallel, speeding up the process
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Callable<DecisionTree>> trainingTasks = new ArrayList<>();

        for (int i = 0; i < numTrees; i++) {
            final int treeIndex = i;
            trainingTasks.add(() -> {
                System.out.println("  Training tree " + (treeIndex + 1) + "/" + numTrees);
                // Bootstrap aggregation (Bagging): Create a sample of the training data
                // by sampling with replacement. Each tree sees a slightly different subset.
                List<MnistImage> bootstrapSample = new ArrayList<>(trainingData.size());
                for (int j = 0; j < trainingData.size(); j++) {
                    bootstrapSample.add(trainingData.get(random.nextInt(trainingData.size())));
                }

                // Create a new DecisionTree for this bootstrap sample
                // Each tree gets its own Random instance, initialized with a unique seed
                // derived from the main Random, to ensure independent randomness.
                DecisionTree tree = new DecisionTree(maxDepth, minSamplesSplit, numFeatures, numRandomFeatures, new Random(random.nextLong()));
                tree.train(bootstrapSample); // Train the individual tree
                return tree;
            });
        }

        // Submit all tasks and wait for them to complete
        List<Future<DecisionTree>> futures = executor.invokeAll(trainingTasks);
        executor.shutdown(); // Shut down the executor after tasks are submitted

        // Collect the trained trees
        for (Future<DecisionTree> future : futures) {
            trees.add(future.get()); // .get() will block until the tree is trained
        }
        System.out.println("Random Forest training complete.");
    }

    @Override
    public int predict(double[] features) throws Exception {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Random Forest not trained.");
        }
        if (features.length != numFeatures) {
            throw new IllegalArgumentException("Input features dimension mismatch. Expected " + numFeatures + " got " + features.length);
        }

        // Collect predictions from all individual trees
        List<Integer> predictions = new ArrayList<>();
        for (DecisionTree tree : trees) {
            predictions.add(tree.predict(features));
        }

        // Majority voting: The final prediction is the class that most trees predicted
        Map<Integer, Long> classCounts = predictions.stream()
                .collect(Collectors.groupingBy(p -> p, Collectors.counting()));

        return classCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue()) // Find the entry with the maximum count
                .map(Map.Entry::getKey)            // Get the class label (key)
                .orElse(-1);                       // Fallback, should not be reached if predictions list is not empty
    }
}