package com.example.mnistpredictor.demo.ml;

import com.example.mnistpredictor.demo.model.MnistImage;
import java.util.List;

public interface Classifier extends Model {
    void train(List<MnistImage> trainingData) throws Exception;
    public int predict(double [] features) throws Exception; 
}
