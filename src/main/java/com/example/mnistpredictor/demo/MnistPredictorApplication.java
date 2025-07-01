package com.example.mnistpredictor.demo;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MnistPredictorApplication implements CommandLineRunner {

    public static void main(String[] args) {
        // Disable web environment since we're not running a web server
        System.setProperty("spring.main.web-application-type", "none");
        SpringApplication.run(MnistPredictorApplication.class, args);
    }

    @Override
    public void run(String... args) throws Exception {
        System.out.println("=".repeat(80));
        System.out.println("MNIST MODEL TRAINING AND EVALUATION");
        System.out.println("=".repeat(80));
        
        // Run the enhanced model training and evaluation
        ModelStatsRunner.main(args);
        
        System.out.println("=".repeat(80));
        System.out.println("TRAINING AND EVALUATION COMPLETE");
        System.out.println("Check the 'results/' directory for detailed evaluation reports.");
        System.out.println("=".repeat(80));
    }
}