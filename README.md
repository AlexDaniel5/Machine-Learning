### General Information
Author: Alex Daniel
Date: Decemeber 8, 2024git 
To Run: python main.py

## Libraries
- Scipy
- Numpy
- Matplotlib
- Pandas
- Scikit-learn

## Syntax Links
- train_test_split: https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.train_test_split.html
- LogisticRegression(): https://realpython.com/logistic-regression-python/
- LinearDiscriminantAnalysis(): https://scikit-learn.org/dev/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
- KNeighborsClassifier(): https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
- DecisionTreeClassifier(): https://www.datacamp.com/tutorial/decision-tree-classification-python

## Algorithm Spot Checks
- Logisitic Regression: Determines if the multi-variate sample (x-axis) can be used in relation to determine the class (y-axis). Similar, to the one line that Victoria showed on the graph with mutliple coloured dots. I used the default arguement which works best when there is regularization (less noise in the column)

- Linear Discriminant Analysis: Determines how to best seperate each variable (column) reducing dimensionality. LDA determines how to seperate classes determined by a mean-vector, spread of data points, spread between class means, and uses eignvalues and eigenvectors to best determine which direction maximizes the class seperation. If the value is high, it may be wise to use a confusion matrix (sklearn.metrics.confusion_matrix) to best determine which classes are being predicted correctly and which are being misclassified

- KNeighborsClassifier(): Attempts to predict the next datapoint within a class. Uses Euclidian distance and other metrics like Manhatten or Minkowski (algorithms studied in CIS*3490) with a default k value of 5 to determine the closest k values, repeating this pattern until the data points are done.

- DecisionTreeClassifier(): Also known as the Classification and Regression Trees (CART) algorithm, it recursively splits the dataset into subsets until no further splits can be made. The splits are based on Gini Impurity (the default), which aims to create homogeneous subsets (e.g., the left split may contain all plants shorter than 7cm, while the right split contains those taller or equal to 7cm). At the end of the process, the tree consists of leaf nodes representing class labels (e.g., "yes" or "no"). During prediction, the algorithm traverses the tree, following the splits until it reaches a leaf node, which provides the predicted class. After training, model evaluation assesses how accurately CART splits the data and uncovers patterns. If successful, visualizations such as decision tree diagrams, feature importance charts, or confusion matrix heatmaps are useful for further interpretation.

- GaussianNB(): Naive Bayes assumes that each feature (or column) is independent of the others, even if they might influence each other in reality. The model predicts the class of a new data point based on the column with the highest probability, which is typically determined by the feature with the lowest standard deviation and the mean closest to the data point. Gaussian Naive Bayes (GNB) specifically assumes that each feature follows a Gaussian (normal) distribution. If the model’s accuracy is low, it likely indicates that the data does not conform to a Gaussian distribution, which could affect the model's performance.

- SVC(): The Support Vector Classifier (SVC) treats each column in the dataset as a dimension in a multi-dimensional space, with two columns forming a two-dimensional space. The goal of SVC is to find a linear boundary that separates the data into two classes, with one side representing Class A and the other Class B. In cases where the data is not linearly separable, SVC uses the kernel trick to map the data into a higher-dimensional space, though this comes at the cost of computational efficiency. Importantly, the higher-dimensional space is not physically created; instead, it is simulated using the dot product of the data points. Once the model is trained, SVC predicts the class of new data points based on which side of the hyperplane they fall. SVC incorporates a regularization parameter, C, which plays a crucial role in the model's behavior. A small C is useful for noisy data, as it allows for some misclassification and helps prevent overfitting, while a large C forces the model to fit the training data more strictly, potentially causing overfitting. SVC also includes the gamma parameter, which controls the spread of the kernel function. A large gamma causes the kernel to have a wider influence on the decision boundary, resulting in a smoother, less complex decision boundary. On the other hand, a smaller gamma makes the influence more localized, leading to a more complex decision boundary that fits the training data more tightly. By default, gamma is calculated based on the dataset, automatically determining whether a smaller or larger gamma is more appropriate for the problem at hand.

## Notes
- Low standard deviation in the result output indicates the model is consistent across different folds of cross-validation
- random_state=1: Ensures the datapoints randomly chosen are always the same
- liblinear: Uses a solver best for regularization
- Solver: Adjusts the coefficients used in the equation to solve supervised learning algorithms
- Gaussian Distribution: A distribution where most of the values cluster around the mean

### Description
Randomly 80% of the datapoints are chosen to train the model. Then 20% of the data will be used to validate and evaluate the model's performance after training. We then use several different algorthim spot checks to best determine how to move forward with the data in how the modeling should be done.
