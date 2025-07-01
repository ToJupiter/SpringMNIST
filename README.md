# MNIST in Pure Java

Pure Java-based machine learning models (KMeans, Logistic Regression, Random Forest) on the MNIST dataset. Say no to external machine learning libraries. Also include REST API for prediction.

## How to run

### Prerequisites
- Java 17 or later
- Maven 3.6+

1. Download the data from: [Kaggle for MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
2. Extract the dataset and put the 4 files:
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`

   in the folder: `src/main/resources/data`

### Build the Project and start the application

Open a terminal in the project root and run:

```
mvn clean package
```

To start the Spring Boot application:

```
mvn spring-boot:run
```

### REST API

The REST API is available at `http://localhost:8080/` (see `PredictionController` for endpoints).

## Results

Model results and comparisons are saved in the `results/` directory after running experiments.

