# A Boosted Trees implementation for Spark/MLLib

This is the working document for an implementation of decision trees
and boosting machine learning algorithms on Spark/MLLib, i.e., `mllib`.
The implementation follows a similar design as that of other
machine learning algorithms currently in `mllib`.
Since decision trees and boosting are primarily meant for
regression and classification, it closely follows the design
of linear and logistic regression currently in `mllib`.
A notable difference is a new `mllib.loss` package
that defines an interface for loss functions,
and is akin to the `mllib.optimization` package
that is primarily used for linear models. 
The decision tree algorithm is implemented using this loss interface
and can thus support any loss function that is implemented using it.
The regression tree and classification tree algorithms simply 
invoke the decision tree algorithm with
square loss and entropy loss respectively.
Details of the design are discussed in the
[Design and Implementation](#design-and-implementation) section.

## Top-level programs

The training data file(s) should be in the same labeled-point format
as that used by linear and logistic regression,
consisting of one instance per line and each line consisting
of the instance label, followed by a comma,
followed by the feature values separated by spaces, i.e.,
`<label>,<feature_1_value> <feature_2_value> ... <feature_p_value>`.
Labels are real numbers for regression and in `{0,1}` for classification.
The feature values are also real numbers.

The implementation supports categorical features.
However, they should be encoded as integers prior to the training procedure.
This processing currently needs to be handled outside of the training programs.
The encoded values should be preferably in the range
`{0,1,...,K-1}` in arbitrary order, where `K` is
the number of categories of the feature.
As an example, if a categorical feature takes values
{`apple`, `orange`, `banana`}, one may encode using
{`apple -> 1`, `orange -> 2`, `banana -> 0`} in the dataset.

If the dataset has categorical features, a header file needs to provided,
that contains the names of the features, one per line,
with categorical feature names terminated with `$` character.
Actual names do not matter.
The first line corresponds to the label, and the remaining lines
for features, in the same order they are present in the training instances.
Thus, if there are three features in the dataset, first one continuous
and second one categorical, a header file may contain

    Y
    X0
    X1$
    X2

.

If no header file is provided, all features are considered to be continuous.

### Synthetic data generation

A synthetic data generation program is provided that
is similar to existing ones for other learning algorithms.
It generates a classification dataset with labels `0` or `1`
and with two features `X0` continuous and `X1$` categorical.
`X0` takes values uniformly in `[0,1]` range.
`X1` is independent of `X0` and takes values in `{0,1,2}`
with equal likelihoood `1/3`.
The underlying model is `Y = 1(X0 >= 0.5 && X1 == 1)`.
Basic usage is similar to other generators,

    DecisionTreeDataGenerator <master> <output_dir> [num_examples] [num_partitions]

, e.g.,

    time \
    SPARK_CLASSPATH=${JARS}/spark-mllib_2.9.3-0.8.0-incubating.jar \
    ${SPARK_HOME}/spark-class \
    org.apache.spark.mllib.util.DecisionTreeDataGenerator \
      local[4] \
      ${WORK}/data.txt \
    2>spark_log.txt

. The header needs to be generated separately, e.g.,

    echo -e "Y\nX0\nX1$" > ${WORK}/header.txt

.

### Regression tree

The basic usage of the training program `RegressionTree` is

    RegressionTree <master> <input_data_file> <input_header_file> <model_output_dir> <max_depth> <min_gain_fraction> [<use_global_quantiles>]

. E.g.,

    time \
    SPARK_CLASSPATH=${JARS}/spark-mllib_2.9.3-0.8.0-incubating.jar \
    ${SPARK_HOME}/spark-class \
    org.apache.spark.mllib.regression.RegressionTree \
      local[4] \
      ${WORK}/train_data.txt \
      ${WORK}/header.txt \
      ${WORK}/rtree \
      5 \
      0.01 \
    2>spark_log.txt

. The output folder `${WORK}/rtree` contains a summary explanation
of the tree in `tree.txt` as well as the tree model in JSON format
in `tree.json` that can be loaded and used for prediction for testing/runtime.

A test program `RegressionTreeTest` is provided with basic usage as

    RegressionTreeTest <master> <input_data_file> <model_file> <error_dir>

. E.g.,

    time \
    SPARK_CLASSPATH=${JARS}/spark-mllib_2.9.3-0.8.0-incubating.jar \
    ${SPARK_HOME}/spark-class \
    org.apache.spark.mllib.regression.RegressionTreeTest \
      local[4] \
      ${WORK}/test_data.txt \
      ${WORK}/rtree/tree.json \
      ${WORK}/rtree/error \
    2>spark_log.txt

. The test data should in the same format as train data, with expected labels.
The output folder `${WORK}/rtree/error` contains a file `error.txt`
with RMSE and MAE of predictions w.r.t. expected labels.

### Classification tree

The classification programs
`org.apache.spark.mllib.classification.{ClassificationTree, ClassificationTreeTest}`
have the same usage, except that the training data
should have labels `0` or `1`. In fact, one can run `RegressionTree`
to train a regression tree model (square loss) on such a dataset
and evaluate error using `ClassificationTreeTest`.
The error statistics for `ClassificationTreeTest` are
TPR (recall), FPR, precision, F-score and accuracy.


## Design and Implementation

### Loss interface

The loss interface is designed for use in a decision tree algorithm
implementation that is efficient and uniform across various loss functions.
The tree growing algorithm is a recursive procedure that
involves finding the best split of the training data
in terms of one of the features so that the labels in each part have
low variance/impurity, and repeating the process for each of the data subsets.
Finding the best split for a particular feature involves dividing
the data instances into bins according to the values of this feature in
the instances. (The bins are defined by quantiles for continuous features
and by the categories themselves for a categorical feature.)
Thereafter, the loss for candidate splits, defined by various groupings
of these bins, needs to be evaluated.

For several loss functions, this can be done efficiently as follows.
There exist summary statistics of the bins such that the
variance/impurity of a group of bins can be calculated
from the statistics obtained by merging/adding those of the constituent bins.
As an example, for square loss, one such list of statistics consists
of the count of values, the sum of values, and the sum of
squares of values, i.e., the zeroth, first and second moments.
These statistics are additive, the centroid is the average,
i.e., `sum/count`, and the variance around centroid is
`sum_square - sum * sum / count`.
Likewise for binary entropy loss and most losses
used for binary classification, one such statistics are
the count of values and the sum of values, i.e., the number of `1`s
if the labels are assumed to be `0` or `1`.
For brevity, we refer to such statistics for various bins
as loss statistics histograms, or moment histograms for square loss.

Based on this observation, the loss interface consists of `LossStats`,
additive summary statistics of a set of labels from which
its variance/impurity w.r.t. the loss function can be calculated.
The `Loss` interface itself contains methods to evaluate the losses
using these summary statistics.
Additional programming details are in the API documentation
of `org.apache.spark.mllib.loss.Loss` and `org.apache.spark.mllib.loss.LossStats`.

### Decision trees

The generic decision tree algorithm implementation
`org.apache.spark.mllib.regression.DecisionTreeAlgorithm`
uses the above loss interfaces. It follows a similar interface
as other algorithms in `mllib`. It is initialized by specifying
various model training parameters, and has a `train` method
that takes the training dataset as input, in the form of an
RDD of labeled points, and outputs a model.
The model is of form `org.apache.spark.mllib.regression.DecisionTreeModel`
and confirms to the `RegressionModel` interface with various predict methods.
Additional methods are provided by the model to aid in saving/loading
a trained model for further use and providing a summary
explanation of the model.

The regression and classification tree algorithms,
`RegressionTreeAlgorithm` and `ClassificationTreeAlgorithm`
are simply derived from `DecisionTreeAlgorithm`
using implementations of square loss and entropy loss
according to the above loss interface.
Both produce a model of form `DecisonTreeModel`.
For classification, the prediction of the model
is interpreted as the likelihood of `1` and needs
to be thresholded (say at `0.5`) for obtaining binary predictions.

## Tests

In addition to the test programs `RegressionTreeTest`
and `ClassificationTreeTest`, functional tests are provided
in `RegressionTreeSuite` and `ClassificationTreeSuite`.
Both tests are performed on data generated by `DecisionTreeDataGenerator`
using the model `Y = 1(X0 >= 0.5 && X1 == 1)` described earlier.

The implementation has been anecdotally tested
to generate the same model as the `rpart` package in `R`
for `auto-mpg` dataset from archives of UCI machine learning repository.
A processed version of this dataset is provided in `mllib/data/tree-data`,
where the 400 instances are separated into 300 train and 100 test instances,
the `make` categorical feature has been encoded (dictionary is included),
and the label, `mpg` has been converted to binary using threshold 25.
Note that there is a significant bias in the original data due to the
instances being arranged in the order of `year`.

## Benchmarks

Tested on a web-scale dataset of ad latency measurements
of size ~ 100 GB with > 1 billion instances and 25 features.
Most of the features are categorical with > 1000 categories,
and some with > 50,000 categories.
Setup involved a Hadoop2/Yarn cluster with 200 workers/containers
set to 8 GB of RAM limit.
Takes 15 minutes to train a classification tree model
and 30 minutes to train a regression tree model,
both limited to depth 5 and minimum gain fraction of 0.01.
Most time is spent on the distributed process of finding
loss statistics histograms of data subsets
indicating a reasonably efficient implementation
in terms of cluster idle time.
The logs can be parsed for info messages from RegressionTree
and ClassificationTree to observe times taken for various steps.

## Todo

 1. Include implementation of loss statistics histograms using `Array`s.
    Investigate why it provides better performance compared to `reduceByKey`,
    i.e., map-reduce shuffle.

 2. Complete `DataEncoder` implementation.

 3. Comprehensive tests, example datasets and additional large-scale benchmarks.

 4. Boosting and ensemble models based on decision trees and other learners.
