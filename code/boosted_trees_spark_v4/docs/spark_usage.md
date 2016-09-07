# Usage on a Standard Spark Setup

The examples show usage of the Spark programs on a standard setup, e.g.,
single-machine local mode or standalone cluster mode.
See the main [README](../README.md) for preliminaries and
[basic usage](../README.md#spark-programs-for-distributed-training).

## Detailed Usage

 1. **Decision tree model trainer**: while most modeling and algorithm
    parameters should be self-explanatory in this example,
    additional explanation is provided below.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkDecisionTreeModelTrainer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --model-dir ${DIST_WORK}/tree \
        --loss-function square \
        --max-depth 5 \
        --min-gain-fraction 0.01 \
        --min-local-gain-fraction 1 \
        --min-count 2 \
        --min-weight 2.0 \
        --feature-weights-file "" \
        --use-sample-weights 0 \
        --use-cache 1 \
        --histograms-method array-aggregate \
        --use-global-quantiles 1 \
        --fast-tree 1 \
        --batch-size 16 \
        --num-reducers-per-node 0 \
        --max-num-reducers 0 \
        --wait-time-ms 1 \
        --use-encoded-data 0 \
        --encoded-data-file ${DIST_WORK}/encoding/encoded_train_data.txt \
        --dicts-dir ${DIST_WORK}/encoding/dicts \
        --save-encoded-data 0 \
        --max-num-quantiles 1000 \
        --max-num-quantile-samples 100000 \
        --rng-seed 42 \
        2>/tmp/spark_log.txt

     Options for `--loss-function` currently include `square` (default)
     for both regression and binary classification,
     and `entropy` for binary classification.
     A common way of generating classification dataset from
     a regression dataset is by mapping the outputs to `0` or `1`
     based on a thresold. See the [utility programs](#utility-programs) section
     for such a program.

     In addition to the regularization options `--max-depth`
     and `--min-gain-fraction`, other options include
     `--min-local-gain-fraction`, `--min-count` and `--min-weight`.
     Even if the gain is less than the `min-gain-fraction` of sample
     variance of full dataset, a split is considered if the gain is greater than
     `min-local-gain-fraction` of variance of samples corresponding to
     the current node. A node is not considered for split if the
     number of samples is less than `min-count` or their total weight
     is less than `min-weight`. If sample weights are not used,
     each sample has weight `1.0`.

     If `--feature-weights-file` is non-empty, weights are assigned to features
     as provided in this file. This file should contain the weights of various
     features, one per line, in the same order as they are present in the
     header file. The first weight in the file, corresponding to the label,
     is ignored. The gains for splits of various features are scaled by their
     respective weights. Thus, features with weights `0.0` are effectively
     ignored. The default feature weights are `1.0`.
     See the [Scala utility programs](scala_usage.md#utility-programs) for
     a convenient way to generate a `0-1` feature weights file from
     input files specifying features to include and exclude.
     Note that the moment histograms and split gains are still calculated
     for features with zero weights.  A better way to ignore features is by
     appending `#` character at the end of their names in the header file.

     If `--use-sample-weights` is `1`, the last field in the data file
     is assumed to be numeric and used as sample weights.
     It is common in binary classification to use different weights
     for positive or negative examples, and likewise, use different
     weights for different output ranges for regression model training.
     See the [utility programs](#utility-programs) section for
     a convenient way to generate a training data file with
     sample weights, based on a unweighted data file and
     steps file that contains the weights to use for various output ranges.

     Apart from these modeling parameters, there are various
     algorithm parameters that affect the performance,
     in terms of training time and/or error.
     If `--use-cache` is `1`, dataset is cached in memory (and disk)
     for substantial speed up in read times.

     The `--histograms-method` is `array-aggregate` by default -
     arrays are used for storing histograms and `mapPartitions` method
     of RDD is used to compute histograms of the data partitions,
     merging them at the master. There is no data shuffle between
     workers in this case (similar to `reduceByKeyLocally`),
     but arrays will use larger memory.
     Alternatively, one may specify `map-reduce`, which uses the `flatMap`
     and `reduceByKey` methods of `RDD` for computing moment histograms.
     A middle-ground is the `aggregate` method. It uses
     `AppendOnlyMap`, an efficient hash-map for aggregation purposes,
     unlike arrays in `array-aggregate`. The part histograms are
     merged by a shuffle operation similar to `reduceByKey`.

     If `--global-quantiles` is `1`, the quantiles for continuous features
     are computed once for the full dataset and used for splitting all nodes,
     instead of computing them each time for the data subset
     corresponding to each node.

     If `--fast-tree` is `1` the histograms for several nodes of a level
     are computed at once, i.e., in a single pass over the dataset.
     The tree is effectively grown level-by-level, instead of node-by-node,
     where each level is trained in smaller batches.
     The `--batch-size` parameter determines the maximum number
     of nodes to train per batch, only used when `--fast-tree` is `1`.
     The number of Spark stages is therefore same as the number of batches,
     instead of the number of nodes in the tree.
     This greatly reduces the training time,
     but the master and workers should each have
     enough memory to hold the histograms for all nodes of a batch at once.
     Depending on the size of the histogram, i.e.,
     the number of distinct feature-value combinations,
     the batch size can be set appropriately.
     The extreme case of `--batch-size 1` is equivalent to `--fast-tree 0`.
     At another extreme, for trees of depth up to 7 and
     less than 100,000 feature-value combinations (e.g., < 100 features,
     each with < 1000 category values or quantile bins),
     one may set batch size to 128, resulting in just one Spark stage per level.

     For `map-reduce` and `aggregate` histogram methods,
     the parameters `--num-reducers-per-node` and `max-num-reducers`
     specify the number of reduce tasks to use for shuffle operations per node,
     up to a maximum number per stage. If these are set to `0`,
     the default settings in Spark are used, namely the number of reduce tasks
     per stage is same as the number of parts in the data file.
     If the number of parts is too large, setting number of reducers per node
     to between 1 and 8, and maximum number of reducers to between 1 and 64
     may result in lower running times.

     The parameter `--wait-time-ms` specifies the time in milliseconds
     to wait initially for workers to be allocated.
     For example, `60000` is a reasonable value for `100` workers
     in a busy cluster.

     Lastly, there are a set of options about indexing categorical features
     in the data. Before the actual model training,
     for each categorical feature, it's `K` values are indexed
     and mapped to numbers in the range `{0,1,...,K-1}`.
     This encoding is performed transparently when using the training programs
     and the generated models do not rely on this encoding.
     However, since it takes a non-trivial
     amount of time to perform the indexing and encoding,
     it may be worthwhile to save the indexing dictionaries for the features
     and the encoded training data for repeated model training.
     If `--use-encoded-data` option is `1`, the encoded training
     data file specified by `--encoded-data-file` is used
     along with the dictionaries specified by `--dicts-dir`.
     The `--data-file` is not used in this case.
     If `--use-encoded-data` option is `0`, the `--data-file` is first indexed,
     and optionally, if `--save-encoded-data` is `1`,
     the dictionaries and encoded data are saved
     to locations specified by `--dicts-dir` and `--encoded-data-file`.
     Also see the [utility programs](#utility-programs) section for
     a program that only performs the indexing.

     Similar to dictionary generation for categorical features,
     quantiles are generated for each of the continues features
     during the indexing phase. The parameter `--max-num-quantiles`
     specifies the maximum number of quantiles to use.
     Quantiles with same value are considered as a single value.
     The quantiles are computed on a sampling of the dataset and
     `--max-num-quantile-samples` specifies the number of samples to use.
     The parameter `--rng-seed` specifies the seed to use for
     random number generation.

 2. **Decision tree error analyzer**:

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkDecisionTreeErrorAnalyzer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/tree \
        --error-dir ${DIST_WORK}/tree/error \
        --binary-mode 0 \
        --threshold 0.5 \
        --full-auc 0 \
        --max-num-summary-samples 100000 \
        --use-cache 0 \
        --rng-seed 42 \
        --wait-time-ms 1 \
        2>/tmp/spark_log.txt

     If `--binary-mode` is `1`, binary classification model and data
     is assumed. The model predictions are considered to be likelihood
     of label to be `1` and labels are generated based on specified
     `--threshold` parameter. In this case, additional error performance metrics
     specific to binary classification, like TPR (Precision), FPR, recall,
     F-score and AUC are generated.

     Scatterplot data consisting of predicted scores and actual labels
     is generated for both regression and classification.
     It can be used for scatterplots in regression problems and
     AUC calculations for classification.
     If `--full-auc` is `1`, AUC is calculated on full data,
     which may take time. This may be done separately using
     the `SparkAUCCalculator` utility program.
     Apart from the full scatterplot data, a sample of the scatterplot data
     is also taken and AUC calculations are performed on it.
     The number of samples can be controlled using `--max-num-summary-samples`.
     The master should be able to hold these samples since
     it does the computation.
     The training data can be cached by setting `--use-cache` to `1`
     for a minor speed-up.
     
     The scatterplot and ROC data for regression and classification
     can be plotted using the respective Python scripts
     [scatter_plot.py](../scripts/scatter_plot.py) and
     [roc_plot.py](../scripts/roc_plot.py) in the [scripts](../scripts) folder.
     E.g., `python scatter_plot.py tree/error/scatter_plot.txt tree/error/scatter_plot.pdf`
     and `python roc_plot.py tree/error/roc.txt tree/error/roc.pdf`.

 3. **GBDT model trainer**:

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkGBDTModelTrainer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --model-dir ${DIST_WORK}/forest \
        --loss-function square \
        --num-trees 5 \
        --shrinkage 0.8 \
        --max-depth 4 \
        --min-gain-fraction 0.01 \
        --min-local-gain-fraction 1 \
        --min-count 2 \
        --min-weight 2.0 \
        --feature-weights-file "" \
        --use-sample-weights 0 \
        --use-cache 1 \
        --histograms-method mapreduce \
        --use-global-quantiles 1 \
        --fast-tree 1 \
        --batch-size 16 \
        --num-reducers-per-node 0 \
        --max-num-reducers 0 \
        --persist-interval 1 \
        --wait-time-ms 1 \
        --use-encoded-data 0 \
        --encoded-data-file ${DIST_WORK}/encoding/encoded_train_data.txt \
        --dicts-dir ${DIST_WORK}/encoding/dicts \
        --save-encoded-data 0 \
        --max-num-quantiles 1000 \
        --max-num-quantile-samples 100000 \
        --rng-seed 42 \
        2>/tmp/spark_log.txt

    Most additional parameters have the same meaning as that
    for decision tree model trainer. The `--persist-interval` parameter
    corresponds to the number of trees that are trained before
    the residual data is cached again. The Spark stages start from the
    RDD corresponding to the last cached residuals.
    Thus, if the interval is too large, the chain of RDDs
    for latter stages are too long and takes larger time to compute.
    However, data is read from disk every time the residuals
    are cached, which can be slow. Thus, one needs to select
    a suitable caching interval to balance this tradeoff.
    Typically a number between 5 to 10 is suitable.

 4. **GBDT error analyzer**:

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkGBDTErrorAnalyzer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/forest \
        --error-dir ${DIST_WORK}/forest/error \
        --binary-mode 0 \
        --threshold 0.5 \
        --full-auc 0 \
        --max-num-summary-samples 100000 \
        --use-cache 0 \
        --rng-seed 42 \
        --wait-time-ms 1 \
        2>/tmp/spark_log.txt

    All additional parameters have the same meaning as that
    for decision tree error analyzer.

## Utility Programs

 1. **Data sampler**: takes a small sample fraction of a larger dataset.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkDataSampler \
        ${JARS}/boosted_trees_spark-spark.jar \
        --data-file ${DIST_WORK}/data.txt \
        --sample-data-file ${DIST_WORK}/sample_data.txt \
        --sample-fraction 0.01 \
        --rng-seed 42 \
        2>/tmp/spark_log.txt

 2. **Binary data generator**: converts a regression dataset into a binary
    classification dataset by thresholding response values to 0 and 1 based on
    a given threshold.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkBinaryDataGenerator \
        ${JARS}/boosted_trees_spark-spark.jar \
        --data-file ${DIST_WORK}/data.txt \
        --binary-data-file ${DIST_WORK}/binary_data.txt \
        --threshold 0.5 \
        2>/tmp/spark_log.txt

 3. **Train-test splitter**: partitions the data uniformly at random
    into train and test sets with sizes governed by given `--train-fraction`.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkTrainTestSplitter \
        ${JARS}/boosted_trees_spark-spark.jar \
        --data-file ${DIST_WORK}/data.txt \
        --train-data-file ${DIST_WORK}/split/train_data.txt \
        --test-data-file ${DIST_WORK}/split/test_data.txt \
        --train-fraction 0.8 \
        --rng-seed 42 \
        2>/tmp/spark_log.txt

 4. **Data encoder**: indexes the categorical feature values observed
    in the train set to integer values and encodes the training data
    using the generated indexes. The indexes can be used by
    the model training programs. For each categorical feature called
    `<feature_name>$`, it generates an index file `<feature_name>_index.txt`
    in the directory specified by `--dicts-dir`.
    One may set the options appropriately to use existing dictionaries
    to encode the data, skipping the generation of indexes,
    or just generate indexes only without further encoding the data.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkDataEncoder \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --dicts-dir ${DIST_WORK}/encoding/dicts \
        --encoded-data-file ${DIST_WORK}/encoding/encoded_train_data.txt \
        --generate-dicts 1 \
        --encode-data 1 \
        --max-num-quantiles 1000 \
        --max-num-quantile-samples 100000 \
        --rng-seed 42 \
        --wait-time-ms 1 \
        2>/tmp/spark_log.txt

 5. **Weighted data generator**: generates a dataset with weights attached
    to training samples using a given training dataset and a weight-steps file.
    The weight-steps file contains the weights that need to be
    assigned for samples in various bins, that are in turn defined by thresholds
    for output values. The thresholds should also be described in this file,
    and interleave the weights. If there are `B` bins defined by
    `B-1` thresholds `(t_1,...,t_{B-1})` for output values, and
    the weights are `(w_1,..., w_B)`, the file should contain

        w_1
        t_1
        w_2
        t_2
        ...
        t_{B-1}
        w_B

    Thus, if the file just contains one line, `1`, then it is equivalent
    to the unweighted case.

    The weights are attached to the data as an extra column,
    i.e., at the end of the lines as an extra field.
    An additional header file is generated that contains an extra feature called
    `sample_weight` and should be used subsequently for model training programs.
    This is not required for prediction for testing/runtime since
    the model doesn't refer to the sample weights explicitly.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkWeightedDataGenerator \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --weight-steps-file ${DIST_WORK}/weight_steps.txt \
        --weighted-data-header-file ${DIST_WORK}/weighted_data_header.txt \
        --weighted-data-file ${DIST_WORK}/weighted_train_data.txt \
        2>/tmp/spark_log.txt

 6. **Decision Tree Scorer**: predicts scores on test data using a decision
    tree model trained using `SparkDecisionTreeModelTrainer` or
    `DecisionTreeModelTrainer`. Input is a decision tree model and 
    test data in the same format as the model training and
    error analysis programs.
    Output is the test data with scores, with an column
    appended to the end consisting of the predicted scores.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkDecisionTreeScorer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/tree \
        --output-file ${DIST_WORK}/test_data_with_scores.txt \
        --binary-mode 0 \
        --threshold 0.5 \
        --num-reducers 10 \
        2>/tmp/spark_log.txt

 7. **GBDT Scorer**: predicts scores on test data using a decision
    tree model trained using `SparkGBDTModelTrainer` or `GBDTModelTrainer`.
    Similar to the decision tree scorer program above.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkGBDTScorer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/forest \
        --output-file ${DIST_WORK}/test_data_with_scores.txt \
        --binary-mode 0 \
        --threshold 0.5 \
        --num-reducers 10 \
        2>/tmp/spark_log.txt

 8. **Classification Error Analyzer**: calculates ROC points, i.e.,
   (FPR, TPR, threshold) tuples, AUC, NLL and RIG.
   Input data is the list of prediction scores and corresponding actual labels.
   If no header file is specified, the first two fields of the input file
   are considered to be the `score` and `label` respectively.
   TPR and FPR are also computed separately for given `threshold`.
   Alternatively, a header file and the score and label fields can be specified.
   Output is the ROC as well as AUC, NLL, RIG and accuracy metrics
   for given threshold.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkClassificationErrorAnalyzer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/scores_labels.txt \
        --output-dir ${DIST_WORK}/error \
        --score-field score \
        --label-field label \
        --threshold 0.5 \
        --num-reducers 10 \
        --wait-time-ms 1 \
        2>/tmp/spark_log.txt
