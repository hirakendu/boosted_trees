# Usage on a Spark-on-YARN Setup

The examples show usage of the Spark programs on a YARN cluster.
It is assumed that the working directory on HDFS is `DIST_WORK`, e.g.,

    export DIST_WORK="hdfs:///user/${USER}/temp/boosted_trees_work/"

. A working directory on local filesystem to fetch the results and analyze
further is assumed to be `WORK`, e.g., `WORK=${HOME}/temp/boosted_trees_work`.
The spark application jar for boosted trees is assumed to be located
in local filesystem at `JARS` location, e.g., `JARS=${HOME}/opt/jars`.
It is assumed that the Spark installation bin containing
`spark-class` launcher script is in path, i.e.,
`PATH=${PATH}:${SPARK_HOME}/bin`.
Lastly, as required by a standard Spark-on-YARN setup, it is assumed that
`SPARK_JAR` is set to the assembly jar containing the class
`spark.deploy.yarn.Client`, e.g.,
`export SPARK_JAR=$(ls $SPARK_HOME/assembly/target/scala*/*.jar)`.

The following cluster settings are assumed in the examples
for the Hadoop job queue name, number of workers, worker memory, master memory:

    export QUEUE=default
    export NUM_EXECUTORS=3
    export EXECUTOR_MEMORY=2g
    export DRIVER_MEMORY=3g

.

For an example run on the [auto-mpg](../data/auto-mpg/) dataset on YARN,
copy it to HDFS:

    # hadoop fs -rm -r -f ${DIST_WORK}
    hadoop fs -mkdir -p ${DIST_WORK}
    hadoop fs -copyFromLocal data/auto-mpg/data.txt \
      data/auto-mpg/split data/auto-mpg/header.txt \
      ${DIST_WORK}
    hadoop fs -ls ${DIST_WORK}

.

## Basic Usage

The following basic usage examples are analogous to the ones in the main
[README](../README.md#spark-programs-for-distributed-training).
Note that when running applications using Spark-on-YARN,
the Spark application class that is run is `org.apache.spark.deploy.yarn.Client`
and the actual application class to be run is passed to `yarn.Client`
using the `--class` parameter. The arguments to the actual application
are passed using `--args` parameters, one for each argument.
All input files should be in HDFS, including header files,
and all output files are written to HDFS.

 1. **Decision tree model trainer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkDecisionTreeModelTrainer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --model-dir ${DIST_WORK}/tree \
        --max-depth 5 \
        --min-gain-fraction 0.01

 2. **Decision tree error analyzer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkDecisionTreeErrorAnalyzer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/tree \
        --error-dir ${DIST_WORK}/tree/error

 3. **GBDT model trainer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkGBDTModelTrainer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --model-dir ${DIST_WORK}/forest \
        --num-trees 5 \
        --shrinkage 0.8 \
        --max-depth 4 \
        --min-gain-fraction 0.01

 4. **GBDT error analyzer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkGBDTErrorAnalyzer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/forest \
        --error-dir ${DIST_WORK}/forest/error

## Detailed Usage

 1. **Decision tree model trainer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
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
        --rng-seed 42

 2. **Decision tree error analyzer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
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

 3. **GBDT model trainer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
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
        --histograms-method array-aggregate \
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
        --rng-seed 42

 4. **GBDT error analyzer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
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
        --wait-time-ms 1

## Utility Programs

 1. **Data sampler**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkDataSampler \
        ${JARS}/boosted_trees_spark-spark.jar \
        --data-file ${DIST_WORK}/data.txt \
        --sample-data-file ${DIST_WORK}/sample_data.txt \
        --sample-fraction 0.01 \
        --rng-seed 42

 2. **Binary data generator**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkBinaryDataGenerator \
        ${JARS}/boosted_trees_spark-spark.jar \
        --data-file ${DIST_WORK}/data.txt \
        --binary-data-file ${DIST_WORK}/binary_data.txt \
        --threshold 0.5

 3. **Train-test splitter**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkTrainTestSplitter \
        ${JARS}/boosted_trees_spark-spark.jar \
        --data-file ${DIST_WORK}/data.txt \
        --train-data-file ${DIST_WORK}/split/train_data.txt \
        --test-data-file ${DIST_WORK}/split/test_data.txt \
        --train-fraction 0.8 \
        --rng-seed 42

 4. **Data encoder**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
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
        --wait-time-ms 1

 5. **Weighted data generator**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkWeightedDataGenerator \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --weight-steps-file ${DIST_WORK}/weight_steps.txt \
        --weighted-data-header-file ${DIST_WORK}/weighted_data_header.txt \
        --weighted-data-file ${DIST_WORK}/weighted_train_data.txt

 6. **Decision Tree Scorer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkDecisionTreeScorer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/tree \
        --output-file ${DIST_WORK}/test_data_with_scores.txt \
        --binary-mode 0 \
        --threshold 0.5 \
        --num-reducers 10

 7. **GBDT Scorer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkGBDTScorer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/forest \
        --output-file ${DIST_WORK}/test_data_with_scores.txt \
        --binary-mode 0 \
        --threshold 0.5 \
        --num-reducers 10

 8. **Classification Error Analyzer**:

        time \
        spark-submit \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkClassificationErrorAnalyzer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/scores_labels.txt \
        --output-dir ${DIST_WORK}/error \
        --score-field score \
        --label-field label \
        --threshold 0.5 \
        --num-reducers 10 \
        --wait-time-ms 1
