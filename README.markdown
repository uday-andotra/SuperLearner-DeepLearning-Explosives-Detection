# Comparing SuperLearner and Deep Learning for Explosives Detection

## Overview

This repository contains a project developed for a Data Mining course (Fall 2024) that compares two advanced machine learning techniques—SuperLearner and Deep Learning—for a critical binary classification task: detecting plastic explosives in airline suitcases using X-ray absorption spectrum data. The goal is to achieve near-perfect classification accuracy due to the high stakes of security applications, while also considering model interpretability. The project includes an R Markdown file with the analysis code and a PDF report summarizing the findings.

## Data

The project uses two datasets:

- **Training Dataset (**`pex23train.RDS`**)**: Contains 23 continuous features representing the X-ray absorption spectrum and a binary target variable (1 for explosives present, 0 for absent).
- **Test Dataset (**`pex23test.RDS`**)**: Used to evaluate the final models.

The training data is split into 80% for training and 20% for validation to assess model performance. The datasets have no missing values but may include some outliers.

## Methods

Two machine learning approaches are compared:

### SuperLearner (Ensemble Method)

- **Base Learners**: Combines 13 models, including:
  - Linear Models: GLM, glmnet
  - Ensemble Models: Random Forest, XGBoost
  - Support Vector Machine (SVM)
  - Non-Linear Models: K-Nearest Neighbors (KNN), Neural Network, Decision Trees, Multivariate Adaptive Regression Splines (MARS), Projection Pursuit Regression (PPR)
  - Discriminant Analysis: Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA)
  - Benchmark: Mean of Y
- **Optimization**: A meta-learner selects the best-performing base learners. A Genetic Algorithm (GA) further optimizes the weights of selected learners (e.g., GLM, SVM, XGBoost) with parameters like population size of 100, mutation rate of 0.3, elitism of 2, and crossover probability of 0.5, achieving optimal results in 10 generations.
- **Evaluation**: Performance is assessed using confusion matrices, misclassification rates, ROC curves, AUC, Precision-Recall curves, and F1 scores.

### Deep Learning (Neural Network)

- **Architecture**: A Multilayer Perceptron (MLP) with:
  - Input layer: 23 features
  - Hidden layers: 64 neurons (ReLU activation), 32 neurons (ReLU activation)
  - Output layer: 1 neuron (sigmoid activation) for binary classification
- **Regularization**: Dropout layers (0.4 and 0.2) to prevent overfitting.
- **Training**: Trained for 50 epochs with a batch size of 32, using binary cross-entropy loss and the Adam optimizer.
- **Evaluation**: Assessed using accuracy, loss, confusion matrices, ROC curves, and additional metrics like Precision, Recall, and F1 Score.

## Results

The performance of both methods was evaluated on the validation set, with the following key metrics:

| Method | Validation Accuracy | Misclassification Rate (Validation) |
| --- | --- | --- |
| SuperLearner (Base) | 100% | 0% |
| SuperLearner (GA-optimized) | 100% | 0% |
| Deep Learning | 99.5% | 0.5% |

- **SuperLearner**:
  - **Base Model**: Achieved 99.87% accuracy on the training set (2 false positives, 1 false negative) and 100% accuracy on the validation set.
  - **GA-optimized Model**: Improved to 99.83% accuracy on the training set (4 false positives) and maintained 100% accuracy on the validation set.
  - **Best Base Learner**: Quadratic Discriminant Analysis (QDA) performed exceptionally well.
- **Deep Learning**:
  - Achieved 99.5% accuracy on the validation set, with 374 true positives, 221 true negatives, 3 false positives, and no false negatives.
  - Slightly lower performance than SuperLearner but still highly accurate.
- **Test Set Predictions**: Generated using the optimized SuperLearner model and saved as `pex23_test_predictions.rds` for external evaluation.

Both methods exceeded 95% accuracy, with the GA-optimized SuperLearner showing a slight edge in accuracy and interpretability.

## Conclusion

The GA-optimized SuperLearner outperformed the Deep Learning model, achieving 100% accuracy on the validation set compared to 99.5% for Deep Learning. Additionally, SuperLearner offers greater interpretability, making it more suitable for critical applications like explosives detection, where understanding model decisions is essential. While Deep Learning demonstrated strong performance and the ability to capture complex patterns, its black-box nature is a limitation in this context. The project highlights the effectiveness of ensemble methods for high-stakes classification tasks.

## Future Enhancements

- **Hyperparameter Tuning**: Further optimize hyperparameters for both SuperLearner and Deep Learning to potentially improve performance.
- **Feature Engineering**: Explore additional feature transformations or selection techniques, particularly for the Deep Learning model.
- **Deep Learning Interpretability**: Apply techniques like SHAP or LIME to enhance the interpretability of the neural network, making it more viable for security applications.

## How to Use

To reproduce the analysis:

1. **Install R and Required Packages**: Install the necessary R packages using:

   ```R
   install.packages(c("SuperLearner", "keras3", "caret", "psych", "GA", "dplyr", "ggplot2", "kernlab", "e1071", "neuralnet", "rpart", "earth", "gam", "MASS"))
   ```
2. **Obtain Datasets**: Ensure you have access to `pex23train.RDS` and `pex23test.RDS`. These may be included in the repository or available through course materials.
3. **Run the R Markdown File**:
   - Open `Project1Final.Rmd` in RStudio.
   - Knit the document to execute the code, which performs exploratory data analysis, model training, evaluation, and generates test set predictions (`pex23_test_predictions.rds`).
4. **Review the Report**:
   - Read `Group Project1 - SuperLearner vs. DeepLearner.pdf` for a detailed summary of the project’s purpose, methods, results, and conclusions.

## References

- Cabrera, J., & McDougall, A. (2002). *Statistical Consulting*. Springer.
- R packages: SuperLearner, keras3, caret, psych, GA, dplyr, ggplot2, kernlab, e1071, neuralnet, rpart, earth, gam, MASS.