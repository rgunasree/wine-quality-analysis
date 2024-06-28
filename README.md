# Wine Quality Prediction Model

### Overview

This project aims to predict wine quality using a RandomForestClassifier. The dataset used is the Wine Quality dataset, which contains various physicochemical properties of wine and their respective quality ratings. The code preprocesses the data, trains a RandomForest model, evaluates its performance, and visualizes the results.

### Dataset

The dataset used is `wine_quality.xlsx`, which contains the following columns:

- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality`

The target variable is `quality`, which is used to create a binary classification column `best quality`.

### Prerequisites

Make sure you have the following libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install them using pip:

```sh
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Steps

1. **Import Necessary Libraries**

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, mean_squared_error
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sb
    ```

2. **Load the Dataset**

    ```python
    df = pd.read_excel('/content/wine_quality.xlsx')
    ```

3. **Explore the Dataset**

    ```python
    print(df.head())
    print(df.info())
    print(df.describe())
    ```

4. **Handle Missing Values**

    ```python
    df.fillna(df.mean(), inplace=True)
    ```

5. **Normalize the Data**

    ```python
    norm = MinMaxScaler()
    df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']] = norm.fit_transform(df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']])
    ```

6. **Create a New Column `best quality`**

    ```python
    df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
    ```

7. **Encode Categorical Variables**

    ```python
    df.replace({'white': 1, 'red': 0}, inplace=True)
    ```

8. **Split the Dataset into Features and Target Variable**

    ```python
    features = df.drop(['quality', 'best quality'], axis=1)
    target = df['best quality']
    ```

9. **Split the Dataset into Training and Testing Sets**

    ```python
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=40)
    ```

10. **Normalize the Data Again**

    ```python
    norm = MinMaxScaler()
    x_train = norm.fit_transform(x_train)
    x_test = norm.transform(x_test)
    ```

11. **Train a RandomForestClassifier Model**

    ```python
    rnd = RandomForestClassifier()
    fit_rnd = rnd.fit(x_train, y_train)
    ```

12. **Predict the Target Variable**

    ```python
    y_predict = rnd.predict(x_test)
    ```

13. **Evaluate the Model**

    ```python
    rnd_score = rnd.score(x_test, y_test)
    print('Score of the model: ', rnd_score)
    ```

14. **Calculate Mean Squared Error and Root Mean Squared Error**

    ```python
    rnd_MSE = mean_squared_error(y_test, y_predict)
    rnd_RMSE = np.sqrt(rnd_MSE)
    print('Mean squared error: ', rnd_MSE)
    print('Root mean squared error: ', rnd_RMSE)
    ```

15. **Print Classification Report**

    ```python
    print(classification_report(y_test, y_predict))
    ```

16. **Create a Heatmap to Visualize Highly Correlated Features**

    ```python
    sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
    plt.show()
    ```

17. **Remove Highly Correlated Features**

    ```python
    print(df.columns)
    df = df.drop('citric acid', axis=1)
    ```

### Results

- The model achieved a score of `0.828125`.
- Mean Squared Error: `0.171875`
- Root Mean Squared Error: `0.414578098794425`
- Classification Report:

    ```
                  precision    recall  f1-score   support
    
               0       0.81      0.82      0.81       147
               1       0.84      0.84      0.84       173
    
        accuracy                           0.83       320
       macro avg       0.83      0.83      0.83       320
    weighted avg       0.83      0.83      0.83       320
    ```

### Visualization

A heatmap was generated to visualize highly correlated features, leading to the removal of the `citric acid` feature.

### Conclusion

The RandomForestClassifier performed well in predicting wine quality, achieving an accuracy of 83%. Further optimization and feature engineering can be done to improve the model's performance.
