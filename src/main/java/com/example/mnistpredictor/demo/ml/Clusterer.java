package com.example.mnistpredictor.demo.ml;

import com.example.mnistpredictor.demo.model.MnistImage;
import java.util.List;

public interface Clusterer extends Model {
    /**
     * Trains the clusterer using the provided data.
     * @param data A list of MnistImage objects for clustering.
     * @throws Exception If an error occurs during training.
     */
    void train(List<MnistImage> data) throws Exception;

    /**
     * Assigns a given input feature vector to a cluster.
     * @param features A double array representing the input features (e.g., normalized pixel data).
     * @return The index of the cluster the features belong to.
     * @throws Exception If an error occurs during clustering.
     */
    int cluster(double[] features) throws Exception;

} 
