import unittest
import numpy as np

from src.linear_regression import LinearRegressionParams, LinearRegressionModel, LinearRegressionEstimator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('UsingLinearRegression').getOrCreate()


Xy = [[1, 2, 3, 18], [3, 4, 5, 30]]
xy_columns = ['x1', 'x2', 'x3', 'label']
df = spark.createDataFrame(Xy, xy_columns)
assembler = VectorAssembler(inputCols=['x1', 'x2', 'x3'], outputCol='features')
transformed_df = assembler.transform(df)


class LinearRegressionParamsTestCase(unittest.TestCase):
    def test_FeaturesCol(self):
        test_value = "test_col"
        lr_params = LinearRegressionParams()
        lr_params.setParams(
            featuresCol=test_value
        )
        self.assertEqual(lr_params.getFeaturesCol(), test_value)

    def test_predictionCol(self):
        test_value = "test_col"
        lr_params = LinearRegressionParams()
        lr_params.setParams(
            predictionCol=test_value
        )
        self.assertEqual(lr_params.getPredictionCol(), test_value)
    
    def test_labelCol(self):
        test_value = "test_col"
        lr_params = LinearRegressionParams()
        lr_params.setParams(
            labelCol=test_value
        )
        self.assertEqual(lr_params.getLabelCol(), test_value)

    def test_maxIter(self):
        test_value = 1000
        lr_params = LinearRegressionParams()
        lr_params.setParams(
            maxIter=test_value
        )
        self.assertEqual(lr_params.getMaxIter(), test_value)

    def test_learningRate(self):
        test_value = 0.07
        lr_params = LinearRegressionParams()
        lr_params.setParams(
            learningRate=test_value
        )
        self.assertEqual(lr_params.getLearningRate(), test_value)
    
    def test_tol(self):
        test_value = 1e-8
        lr_params = LinearRegressionParams()
        lr_params.setParams(
            tol=test_value
        )
        self.assertEqual(lr_params.getTol(), test_value)


class LinearRegressionModelTestCase(unittest.TestCase):
    def test_numFeatures(self):
        model = LinearRegressionModel(coefficients=[0, 0, 0, 0], intercept=0)
        self.assertEqual(model.numFeatures, 4)

    def test_predict(self):
        model = LinearRegressionModel(coefficients=[1, 2, 3], intercept=4)
        actual = model.predict(transformed_df.first())
        expected = 18
        self.assertEqual(actual, expected)

    def test_transform(self):
        model = LinearRegressionModel(coefficients=[1, 2, 3], intercept=4)
        actual_df = model.transform(transformed_df)
        actual = [row.prediction for row in actual_df.head(2)]
        expected = [row.label for row in actual_df.head(2)]
        self.assertListEqual(actual, expected)


class LinearRegressionEstimatorTestCase(unittest.TestCase):
    def test_fit(self):
        X = np.random.standard_normal((1000, 3))
        noise = np.random.standard_normal(1000) * 0.01
        expected_coefficients = [1.5, 0.3, -0.7]
        weights = np.array(expected_coefficients)
        y = np.dot(X, weights) + noise

        x_columns = ['x1', 'x2', 'x3']
        y_cloumn = 'y'
        df = spark.createDataFrame(
            np.column_stack((X, y)).tolist(),
            x_columns + [y_cloumn],
        )

        aggregated_features_column = "features"
        df = VectorAssembler(inputCols=x_columns, outputCol=aggregated_features_column).transform(df)

        linear_regression = LinearRegressionEstimator(labelCol=y_cloumn, maxIter=50) 
        model = linear_regression.fit(df)

        actual = model.coefficients + [model.intercept]
        expected = expected_coefficients + [0]

        self.assertTrue(all(abs(actual_item - expedcted_item) < 0.1 for actual_item, expedcted_item in zip(actual, expected)))


if __name__ == "__main__":
    unittest.main()