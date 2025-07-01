package com.example.mnistpredictor.demo.ml.kmeans;

import com.example.mnistpredictor.demo.ml.Clusterer;
import com.example.mnistpredictor.demo.model.MnistImage;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Map;
import java.util.HashMap;


public class KMeansClusterer implements Clusterer, Serializable{
    private static final long serialVersionUID = 1L;

    private double[][] centroids;
    private int k;
    private int numFeatures;
    private int maxIterations;
    private transient Random random;

    public KMeansClusterer(int k, int numFeatures, int maxIterations){
        this.k = k;
        this.numFeatures = numFeatures;
        this.maxIterations = maxIterations;
        this.random = new Random();
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException{
        in.defaultReadObject();
        this.random = new Random();
    }

    @Override
    public void train(List<MnistImage> data) throws Exception{
        System.out.println("Training K-Means clusterer with K=" + k + "...");
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty.");
        }

        // Extract pixel data (features) for clustering
        List<double[]> featuresList = new ArrayList<>();
        for (MnistImage image : data){
            featuresList.add(image.getPixelData());
        }

        // 1. Initialize centroids: Randomly pick 'k' data points from the dataset as initial centroids.
        centroids = new double[k][numFeatures];
        for (int i = 0; i < k; i++){
            // Ensure we don't pick the same data point multiple times for initial centroids (though for large datasets it's unlikely)
            centroids[i] = Arrays.copyOf(featuresList.get(random.nextInt(featuresList.size())), numFeatures);
        }

        for (int iter = 0; iter < maxIterations; iter++){
            Map<Integer, List<double[]>> clusters = new HashMap<>();

            // 2. Assign each data point to its closest centroid

            for (int i = 0; i < k; i++) clusters.put(i, new ArrayList<>());

            for (double[] features : featuresList){
                int closestCentroidIndex = findClosestCentroid(features);
                clusters.get(closestCentroidIndex).add(features);
            }
            
            // 3. Update centroids: Calculate the new mean for each other
            boolean changed = false;
            for (int i = 0; i < k; i++){
                List<double[]> currentClustersPoints = clusters.get(i);
                if (!currentClustersPoints.isEmpty()){
                    double[] newCentroid = new double[numFeatures];
                    // Sum up all feature values for points in the current cluster
                    for (double[] point : currentClustersPoints){
                        for (int j = 0; j < numFeatures; j++) newCentroid[j] += point[j];
                    }

                for (int j = 0; j < numFeatures; j++) newCentroid[j] /= currentClustersPoints.size();
                if (!Arrays.equals(centroids[i], newCentroid)){
                    centroids[i] = newCentroid; 
                    changed = true;
                }
                else {
                    centroids[i] = Arrays.copyOf(featuresList.get(random.nextInt(featuresList.size())), numFeatures);
                    changed = true;
                }


                }
                
                if (!changed){
                    System.out.println("K-Means converged at iteration" + (iter + 1));
                    break;
                }
            }
        }
        
        System.out.println("K-Means training complete");

    }

     /**
     * Finds the index of the closest centroid for a given feature vector using Euclidean distance.
     * @param features The feature vector to assign.
     * @return The index of the closest centroid (cluster ID).
     */

    private int findClosestCentroid(double[] features){
        double minDistance = Double.MAX_VALUE;
        int closestCentroidIndex = -1;

        for (int i = 0; i < k; i++) {
            double distance = euclideanDistance(features, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroidIndex = i;
            }
        }
        return closestCentroidIndex;
    }

      /**
     * Calculates the Euclidean distance between two feature vectors.
     * @param a First vector.
     * @param b Second vector.
     * @return Euclidean distance.
     */
    
    private double euclideanDistance(double[] a, double[] b){
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += Math.pow(a[i] - b[i], 2);
        return Math.sqrt(sum);
    }

    @Override
    public int cluster(double[] features) throws Exception{
        if (centroids == null || centroids.length == 0) throw new IllegalStateException("K-Means not trained yet.");
        if (features.length != numFeatures) throw new IllegalArgumentException("Input dimension mismatch. Expected " + numFeatures + "got " + features.length);
        return findClosestCentroid(features);
    }

}
