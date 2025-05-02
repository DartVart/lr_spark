import os
import sys

from typing import Any
from tqdm import tqdm
import numpy as np

from pyspark.ml.param.shared import HasMaxIter, HasLearningRate, HasTol
from pyspark.ml.regression import RegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.base import _PredictorParams
from pyspark.ml import Estimator
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


class LinearRegressionParams(_PredictorParams, HasMaxIter, HasLearningRate, HasTol):
    def __init__(self, *args: Any):
        super(LinearRegressionParams, self).__init__(*args)
    
    def setParams(
        self,
        *,
        featuresCol: str = "features",
        labelCol: str = "label",
        predictionCol: str = "prediction",
        maxIter: int = 100,
        learningRate: float = 0.1,
        tol: float = 1e-6,
    ):
        return self._set(
            featuresCol=featuresCol,
            labelCol=labelCol,
            predictionCol=predictionCol,
            maxIter=maxIter, 
            learningRate=learningRate, 
            tol=tol
        )


class LinearRegressionModel(
    RegressionModel,
    LinearRegressionParams
):
    def __init__(self, *, coefficients, intercept):
        super(LinearRegressionModel, self).__init__()
        self.coefficients = coefficients
        self.intercept = intercept

    @property
    def numFeatures(self) -> int:
        return len(self.coefficients)

    def predict(self, row) -> float:
        return float(
            np.dot(self.coefficients, row[self.getFeaturesCol()]) +
            self.intercept
        )

    def _transform(self, dataframe) -> float:
        featuresCol = self.getFeaturesCol()
        labelCol = self.getLabelCol()
        predictionCol = self.getPredictionCol()

        transformed_df = dataframe.rdd.map(
            lambda row: (row[featuresCol], row[labelCol], self.predict(row))
        )

        return dataframe.sparkSession.createDataFrame(transformed_df, [featuresCol, labelCol, predictionCol])


class LinearRegressionEstimator(
    Estimator,
    LinearRegressionParams,
):
    def __init__(
        self,
        *,
        featuresCol: str = "features",
        labelCol: str = "label",
        predictionCol: str = "prediction",
        maxIter: int = 100,
        learningRate: float = 0.1,
        tol: float = 1e-6,
    ):
        super(LinearRegressionEstimator, self).__init__()
        self.setParams(
            featuresCol=featuresCol,
            labelCol=labelCol,
            predictionCol=predictionCol,
            maxIter=maxIter, 
            learningRate=learningRate, 
            tol=tol,
        )

    @staticmethod
    def _get_gradients(row, coefficients, intercept):
        X, y = row
        error = np.dot(coefficients, X) + intercept - y
        return X * error, error

    def _fit(self, dataframe):
        features_col = self.getFeaturesCol()
        prediction_col = self.getPredictionCol()
        label_col = self.getLabelCol()
        learning_rate = self.getLearningRate()
        tol = self.getTol()
        max_iter = self.getMaxIter()

        cashed_dataframe = dataframe.rdd.map(
            lambda row: (
                np.array(row[features_col]), 
                row[label_col]
            )
        ).cache()

        n = cashed_dataframe.count()
        num_features = len(cashed_dataframe.first()[0])

        coefficients = np.ones(num_features)
        intercept = 0

        for epoch in tqdm(range(max_iter)):
            gradients_sum = cashed_dataframe.map(
                lambda row: self._get_gradients(row, coefficients, intercept)
            ).reduce(
                lambda acc_sum, new_gradiends: (acc_sum[0] + new_gradiends[0], acc_sum[1] + new_gradiends[1])
            )

            new_coefficients = coefficients - learning_rate * gradients_sum[0] / n
            new_intercept = intercept - learning_rate * gradients_sum[1] / n

            if np.sqrt(np.sum((new_coefficients - coefficients) ** 2) + (new_intercept - intercept) ** 2) < tol:
                print(f'Fitting stopped at epoch {epoch}.')
                break
            
            coefficients = new_coefficients
            intercept = new_intercept
            
        model = LinearRegressionModel(coefficients=coefficients, intercept=intercept)
        model.setParams(
            featuresCol=features_col,
            labelCol=label_col,
            predictionCol=prediction_col,
            maxIter=max_iter, 
            learningRate=learning_rate, 
            tol=tol,
        )
        cashed_dataframe.unpersist()

        return model
 

if __name__ == '__main__':
    spark = SparkSession.builder.appName('UsingLinearRegression').getOrCreate()

    X = np.random.standard_normal((1000, 3))
    noise = np.random.standard_normal(1000) * 0.01
    weights = np.array([1.5, 0.3, -0.7])
    y = np.dot(X, weights) + noise

    x_columns = ['x1', 'x2', 'x3']
    y_cloumn = 'y'
    df = spark.createDataFrame(
        np.column_stack((X, y)).tolist(),
        x_columns + [y_cloumn],
    )

    aggregated_features_column = "features"
    assembler = VectorAssembler(inputCols=x_columns, outputCol=aggregated_features_column)
    transformed_df = assembler.transform(df)

    print(transformed_df)

    linear_regression = LinearRegressionEstimator(labelCol=y_cloumn, maxIter=50) 
    model = linear_regression.fit(transformed_df)

    print(model.coefficients)
