# Boosted Trees on Spark and Scala

## Build

Pre-requisites:
  1. JDK-1.6 and above
  2. Scala 2.9.3
  3. Maven 3.0.4 and above
  4. Spark 0.8.0-SNAPSHOT from trunk, built with `mvn -Phadoop2-yarn`.
  
The `yarn.version` property should correspond to the same Hadoop 2 version
in the POM files of both `spark` and `boosted_trees_spark`.
This is currently set to the `0.23.8` version of Hadoop.
May also edit the `spark.version` property in the POM file of this project.

Run `mvn install` in the project folder

    cd boosted_trees_spark
    mvn clean install

This should create the jar files `boosted_trees_spark.jar` and
`boosted_trees_spark-spark.jar` in `target/` subfolder.
The first jar file does not include dependencies, while the second one does. 
The second jar can be used in two ways. 
he single-machine, non-distributed programs
can be run using `scala` by including the jar in classpath.
The distributed programs can be run by using it as a _Spark jar_ with
the `run` script of Spark. Although it is built and tested on Spark + Yarn,
the Spark jar should be able to run on any Spark cluster
similar to the method described later on for running on local Spark instance
using the provided script `scripts/spark_local_run.sh`.

## Run

### Single-machine scala programs

The single-machine implementation includes several programs,
i.e., main classes, which can used in a typical workflow
for regression analysis of a dataset using GBRT.
These include sampling from a large dataset,
splitting a dataset into train and test sets,
indexing the categorical features in the training dataset,
training a regression tree or GBRT forest model,
printing verbose details about the model, and
predicting and evaluating error a test dataset.
They can be run using `scala` by including the `boosted_trees_spark-spark.jar`
in classpath. This does not require a Spark installation.

The programs use the directory `${HOME}/temp/boosted_trees_work/`
by default for various working files and folders.
The dataset format primarily consists of two parts:

 1. **Data file**: a plain text file containing one sample per line,
    each sample consisting of tab separated feature values.
    The first field/value in each line is considered to be the _label_.
    The values correspond to continuous (numbers) or
    categorical features (text).
     
 2. **Header/schema file:** a plain text file containing the feature names,
    one feature per line. The first line/feature name is always `label`.
    Feature names ending with the dollar character `$` are considered
    to be categorical.
    
By default, the expected data file is `${HOME}/temp/boosted_trees_work/data.txt`
and header file is `${HOME}/temp/boosted_trees_work/header.txt`.

A sample dataset is provided in `data/auto-mpg` folder along with
modeling results. This is a well known dataset taken from
[UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Auto+MPG).
Pre-processing steps are described in `data/auto-mpg/notes.txt`.
For an example run, copy this to the work folder as follows:

    rm -rf ${HOME}/temp/boosted_trees_work && mkdir ${HOME}/temp/boosted_trees_work
    cp -a data/auto-mpg/{data.txt,header.txt,split} ${HOME}/temp/boosted_trees_work/

In addition, one can optionally assign weights to features and
training samples using the following files:

 3. **Feature weights file (optional):** a plain text file containing feature weights,
    one per line, corresponding to the features in the header file.
    If there are two many features, one can specify features to exclude
    or include and generate a 0-1 weights file using the simple
    feature weights generator program in this package.
    At each split step, the gains of best splits for features
    are scaled by their respective weights.
     
 4. **Weight steps file (optional):** a plain text file describing the weights
    to assign for samples in various bins, in turn defined by thresholds
    for output values. The thresholds should also be described in this file,
    and interleave the weights. Thus, if there are `B` bins defined by
    `B-1` thresholds `(t_1,...,t_{B-1})` for output values, and
    the weights are `(w_1,..., w_B)`, the file should contain

        w_1
        t_1
        w_2
        t_2
        ...
        t_{B-1}
        t_B

    Thus, if the file just contains one line, `1`, then it is equivalent
    to the unweighted case. See the weighted data generator programs for using
    the weight-steps file to attach weights to training samples.

The program names and their parameters are mostly self-explanatory.
Sample commands for running each of these programs and their default
options are provided below with brief explanations.
For examples, we assume that the `${JARS}` folder contains
`boosted-trees_spark-spark.jar` and the working directory is
`${WORK}`, say

    export JARS="${HOME}/jars/"
    export WORK="${HOME}/temp/boosted_trees_work"
.

 1. **Data sampler**: takes a small sample fraction of a big dataset.
    
        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.DataSampler \
          --data-file ${WORK}/data.txt \
          --sample-data-file ${WORK}/sample_data.txt \
          --sample-rate 0.01
     
 2. **Binary data generator**: converts a regression dataset into a binary
    classification dataset by thresholding response values to 0 and 1 based on
    a given threshold.

        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.BinaryDataGenerator \
          --data-file ${WORK}/data.txt \
          --binary-data-file ${WORK}/binary_data.txt \
          --threshold 0.5
     
 3. **Train-test splitter**: partitions the data into train and
    test sets: `${WORK}/split/train_data.txt` and
    `${WORK}/split/test_data.txt`.

        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.TrainTestSplitter \
          --data-file ${WORK}/data.txt \
          --train-data-file ${WORK}/split/train_data.txt \
          --test-data-file ${WORK}/split/test_data.txt \
          --train-fraction 0.8
     
 4. **Data indexer**: indexes the categorical feature values observed
    in the train set to integer values and encodes the training data
    using the generated indexes. The indexes are used later
    on for prediction and test data analysis. For each categorical feature called
    `<feature_name>$`, it generates an index file
    `${WORK}/indexing/indexes/<feature_name>_index.txt`.
    The encoded train data is `${WORK}/indexing/indexed_train_data.txt`.

        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.DataIndexer \
          --header-file ${WORK}/header.txt \
          --data-file ${WORK}/split/train_data.txt \
          --indexes-dir ${WORK}/indexing/indexes/ \
          --indexed-data-file ${WORK}/indexing/indexed_train_data.txt \
          --save-indexed-data 1
     
 5. **Weighted data generator**: generates a dataset with weights attached
    to training samples using a given training dataset and a weight-steps file.
    The weights are attached at the end as an extra field.
    The program can be used before indexing step as well on the raw training data.
    Can also be used as a template for other ways of weighting samples.
    The generated header file that contains an extra feature called `sample_weight`
    should be used subsequently for model training programs.
    This is not required for prediction for testing/runtime since
    the model outputs are not affected explicitly.

        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.WeightedDataGenerator \
          --header-file ${WORK}/header.txt \
          --weighted-data-header-file ${WORK}/weighted_data_header.txt \
          --data-file ${WORK}/indexing/indexed_train_data.txt \
          --weighted-data-file ${WORK}/indexing/indexed_weighted_train_data.txt
              
 6. **Feature weights generator**: to generate a 0-1 feature weights file
    corresponding to features in header file and those specified in the includes
    and excludes file. The excludes are applied after includes.
    Use empty arguments (default) or don't specify the argument
    to include all features.

        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.FeatureWeightsGenerator \
          --header-file ${WORK}/header.txt \
          --included-features-file ${WORK}/included_features.txt \
          --excluded-features-file ${WORK}/excluded_features.txt \
          --feature-weights-file ${WORK}/feature_weights.txt
     
 7. **Regression tree model trainer**: trains a regression tree model
    on training data. Training data can be raw or indexed -- in the former
    case, indexing is performed before training. The output model is saved to
    `${WORK}/indexing/tree/nodes.txt` which is used for prediction,
    for analyzing test error or runtime. An easy-to-read version is printed
    to `${WORK}/tree/tree.txt`.
    The baseline is trivial constant response, the average output of train data,
    and the corresponding RMSE (root of sample variance) and MAE.
    Empty feature weights file or omitted argument indicates no feature weights.
    If using sample weights, use the appropriate data and header file.

        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.RegressionTreeModelTrainer \
          --header-file ${WORK}/header.txt \
          --feature-weights-file "" \
          --data-file ${WORK}/split/train_data.txt \
          --indexes-dir ${WORK}/indexing/indexes/ \
          --indexed-data-file ${WORK}/indexing/indexed_train_data.txt \
          --model-dir ${WORK}/tree/ \
          --max-depth 5 \
          --min-gain-fraction 0.01 \
          --use-sample-weights 0 \
          --use-indexed-data 0 \
          --save-indexed-data 0
     
 8. **Regression tree error analyzer**: predicts on test data using a tree model
    and calculates test error in terms of RMSE (root mean square error)
    and MAE (mean absolute error). Test data can be raw or indexed -- in the former
    case, indexing is performed using same indexes as train data before prediction.
    Reads tree model from `${WORK}/tree/tree.txt`.
    The output errors are saved to `${WORK}/tree/error.txt`.

        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.RegressionTreeErrorAnalyzer \
          --header-file ${WORK}/header.txt \
          --data-file ${WORK}/split/test_data.txt \
          --indexes-dir ${WORK}/indexing/indexes/ \
          --indexed-data-file ${WORK}/indexing/indexed_test_data.txt \
          --model-dir ${WORK}/tree/ \
          --error-file ${WORK}/tree/error.txt \
          --binary-mode 0 \
          --threshold 0.5 \
          --use-indexed-data 0 \
          --save-indexed-data 0
        
 9. **Regression tree details printer**: prints detailed information
    about a tree model. Reads model from `${WORK}/tree/nodes.txt`
    and prints details to `${WORK}/tree/tree_details.txt`
    and `${WORK}/tree/nodes_details/`.
 
        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.RegressionTreeDetailsPrinter \
          --header-file ${WORK}/header.txt \
          --indexes-dir ${WORK}/indexing/indexes/ \
          --model-dir ${WORK}/tree/
     
 10. **Regression tree DOT printer**: prints a Graphviz DOT file that
    illustrates a tree model. Also prints PDF if `graphviz` and `dot` are installed.
    Reads model from `${WORK}/tree/nodes.txt`
    and prints details to `${WORK}/tree/tree.dot`
    and `${WORK}/tree/tree.pdf`.
 
        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.RegressionTreeDotPrinter \
          --header-file ${WORK}/header.txt \
          --model-dir ${WORK}/tree/
             
 11. **GBRT model trainer**: trains a GBRT forest model on training data.
    Most options are similar to regression tree model trainer.
    The output model is saved to `${WORK}/indexing/tree/forest/nodes/`.
    An easy-to-read version is printed to `${WORK}/forest/trees/`.
 
        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.GBRTModelTrainer \
          --header-file ${WORK}/header.txt \
          --feature-weights-file "" \
          --data-file ${WORK}/split/train_data.txt \
          --indexes-dir ${WORK}/indexing/indexes/ \
          --indexed-data-file ${WORK}/indexing/indexed_train_data.txt \
          --model-dir ${WORK}/forest/ \
          --num-trees 5 \
          --shrinkage 0.8 \
          --max-depth 4 \
          --min-gain-fraction 0.01 \
          --use-sample-weights 0 \
          --use-indexed-data 0 \
          --save-indexed-data 0
     
 12. **GBRT error analyzer**: predicts on test data using a forest model
    and calculates errors. Most options are similar to
    regression tree error analyzer.
 
        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.GBRTErrorAnalyzer \
          --header-file ${WORK}/header.txt \
          --data-file ${WORK}/split/test_data.txt \
          --indexes-dir ${WORK}/indexing/indexes/ \
          --indexed-data-file ${WORK}/indexing/indexed_test_data.txt \
          --model-dir ${WORK}/forest/ \
          --error-file ${WORK}/forest/error.txt \
          --binary-mode 0 \
          --threshold 0.5 \
          --use-indexed-data 0 \
          --save-indexed-data 0
         
 13. **GBRT details printer**: prints detailed information
    about a forest model.
 
        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.GBRTDetailsPrinter \
          --header-file ${WORK}/header.txt \
          --indexes-dir ${WORK}/indexing/indexes/ \
          --model-dir ${WORK}/forest/
          
 14. **GBRT DOT printer**: prints Graphviz DOT files that
    illustrates the trees in a forest model.
    Also prints PDF if `graphviz` and `dot` are installed.
 
        time scala -cp ${JARS}/boosted_trees_spark-spark.jar \
          boosted_trees.GBRTDotPrinter \
          --header-file ${WORK}/header.txt \
          --model-dir ${WORK}/forest/

### Distributed Spark programs

The files and folders are to be specified with the protocol
and full path. Both file:// for local and hdfs:// for HDFS locations
are supported by Spark. Any text file written out by Spark using Hadoop
API is in the form of a folder with part files inside it.
Use the script `hadoop_txt_compact.sh` in the `scripts/` folder
to compact them into usual text files.

The working directory in these examples is assumed to be `${DIST_WORK}`, say

    export DIST_WORK="file:///${HOME}/temp/boosted_trees_work/"  # for local
    # or
    export DIST_WORK="hdfs:///user/${USER}/temp/boosted_trees_work/"  # for cluster
.

Depending on the Spark cluster setup, there are different ways of
running the distributed versions of the programs.
The usage is almost similar to the single-machine versions
in terms of arguments. The distributed programs take additional
arguments `spark-master`, `spark-home` and `spark-app-jars`,
which can be also specified using the environment variables
`SPARK_MASTER`, `SPARK_HOME` and `SPARK_APP_JARS`.

A convenient way to run programs is to create a script, say
`spark_run.sh` in `PATH` which `cd`'s into the spark home
directory and runs the `./run` with the provided arguments.
Two such sample scripts are provided in `scripts/` folder,
`spark_local_run.sh` and `spark_yarn_run.sh` for running
locally and on a Hadoop 2 (Yarn) cluster respectively.

Thus, to run the data sampler program locally with
2 workers and 2 cores per worker, the command is:

    time \
    SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
    spark_local_run.sh boosted_trees.spark.SparkDataSampler \
      --spark-master local[2,2] \
      --data-file ${DIST_WORK}/data.txt \
      --sample-data-file ${DIST_WORK}/sample_data.txt \
      --sample-rate 0.01 \
      2>spark_log.txt
  
or as caret separated:

    time \
    SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
    spark_local_run.sh boosted_trees.spark.SparkDataSampler \
    --spark-master^local[2,2]^\
    --data-file^${DIST_WORK}/data.txt^\
    --sample-data-file^${DIST_WORK}/sample_data.txt^\
    --sample-rate^0.01 \
      2>spark_log.txt

. To run on a Hadoop 2 (Yarn) cluster:

    time spark_yarn_run.sh \
      --jar ${JARS}/boosted_trees_spark-spark.jar \
      --class boosted_trees.spark.SparkDataSampler \
      --num-workers 4 \
      --worker-memory 3g \
      --worker-cores 1 \
      --master-memory 3g \
      --queue default \
      --args \
    --spark-master^yarn-standalone^\
    --data-file^${DIST_WORK}/data.txt^\
    --sample-data-file^${DIST_WORK}/sample_data.txt^\
    --sample-rate^0.01

. For running on Yarn cluster, the `spark-master` is to be specified
as 	`yarn-standalone` and the arguments to the program are merged
into a single argument separated by carets `^` and passed as `args`.

Examples of the respective commands for distributed programs
using `spark_local_run.sh` similar to the single machine version
are provided below. For examples of the same on Yarn cluster using
`spark_yarn_run.sh`, see [yarn_examples](docs/yarn_examples.md).

 1. **Data sampler**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkDataSampler \
        --spark-master^local[4]^\
        --data-file^${DIST_WORK}/data.txt^\
        --sample-data-file^${DIST_WORK}/sample_data.txt^\
        --sample-rate^0.01 \
          2>spark_log.txt
     
 2. **Binary data generator**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkBinaryDataGenerator \
        --spark-master^local[4]^\
        --data-file^${DIST_WORK}/data.txt^\
        --binary-data-file^${DIST_WORK}/binary_data.txt^\
        --threshold^0.5 \
          2>spark_log.txt
            
 3. **Train-test splitter**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkTrainTestSplitter \
        --spark-master^local[4]^\
        --data-file^${DIST_WORK}/data.txt^\
        --train-data-file^${DIST_WORK}/split/train_data.txt^\
        --test-data-file^${DIST_WORK}/split/test_data.txt^\
        --train-fraction^0.8 \
          2>spark_log.txt
     
 4. **Data indexer**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkDataIndexer \
        --spark-master^local[4]^\
        --header-file^${DIST_WORK}/header.txt^\
        --data-file^${DIST_WORK}/split/train_data.txt^\
        --indexes-dir^${DIST_WORK}/indexing/indexes/^\
        --indexed-data-file^${DIST_WORK}/indexing/indexed_train_data.txt^\
        --save-indexed-data^1 \
          2>spark_log.txt
     
 5. **Weighted data generator**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkWeightedDataGenerator \
        --spark-master^local[4]^\
        --header-file^${DIST_WORK}/header.txt^\
        --weighted-data-header-file^${DIST_WORK}/weighted_data_header.txt^\
        --data-file^${DIST_WORK}/indexing/indexed_train_data.txt^\
        --weighted-data-file^${DIST_WORK}/indexing/indexed_weighted_train_data.txt \
          2>spark_log.txt
     
 6. **Regression tree model trainer**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkRegressionTreeModelTrainer \
        --spark-master^local[4]^\
        --header-file^${DIST_WORK}/header.txt^\
        --feature-weights-file^^\
        --data-file^${DIST_WORK}/split/train_data.txt^\
        --indexes-dir^${DIST_WORK}/indexing/indexes/^\
        --indexed-data-file^${DIST_WORK}/indexing/indexed_train_data.txt^\
        --model-dir^${DIST_WORK}/tree/^\
        --max-depth^5^\
        --min-gain-fraction^0.01^\
        --min-distributed-samples^10000^\
        --use-sample-weights^0^\
        --use-indexed-data^0^\
        --save-indexed-data^0^\
        --cache-indexed-data^0 \
          2>spark_log.txt
     
 7. **Regression tree error analyzer**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkRegressionTreeErrorAnalyzer \
        --spark-master^local[4]^\
        --header-file^${DIST_WORK}/header.txt^\
        --data-file^${DIST_WORK}/split/test_data.txt^\
        --indexes-dir^${DIST_WORK}/indexing/indexes/^\
        --indexed-data-file^${DIST_WORK}/indexing/indexed_test_data.txt^\
        --model-dir^${DIST_WORK}/tree/^\
        --error-file^${DIST_WORK}/tree/error.txt^\
        --binary-mode^0^\
        --threshold^0.5^\
        --use-indexed-data^0^\
        --save-indexed-data^0 \
          2>spark_log.txt
     
 8. **GBRT model trainer**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkGBRTModelTrainer \
        --spark-master^local[4]^\
        --header-file^${DIST_WORK}/header.txt^\
        --feature-weights-file^^\
        --data-file^${DIST_WORK}/split/train_data.txt^\
        --indexes-dir^${DIST_WORK}/indexing/indexes/^\
        --indexed-data-file^${DIST_WORK}/indexing/indexed_train_data.txt^\
        --residual-data-file^${DIST_WORK}/indexing/residual_data.txt^\
        --model-dir^${DIST_WORK}/forest/^\
        --num-trees^5^\
        --shrinkage^0.8^\
        --max-depth^4^\
        --min-gain-fraction^0.01^\
        --min-distributed-samples^10000^\
        --use-sample-weights^0^\
        --initial-num-trees^0^\
        --residual-mode^0^\
        --use-indexed-data^0^\
        --save-indexed-data^0^\
        --cache-indexed-data^0 \
          2>spark_log.txt
     
 9. **GBRT error analyzer**:

        time \
        SPARK_CLASSPATH=${JARS}/boosted_trees_spark-spark.jar \
        spark_local_run.sh boosted_trees.spark.SparkGBRTErrorAnalyzer \
        --spark-master^local[4]^\
        --header-file^${DIST_WORK}/header.txt^\
        --data-file^${DIST_WORK}/split/test_data.txt^\
        --indexes-dir^${DIST_WORK}/indexing/indexes/^\
        --indexed-data-file^${DIST_WORK}/indexing/indexed_test_data.txt^\
        --model-dir^${DIST_WORK}/forest/^\
        --error-file^${DIST_WORK}/forest/error.txt^\
        --binary-mode^0^\
        --threshold^0.5^\
        --use-indexed-data^0^\
        --save-indexed-data^0 \
          2>spark_log.txt

## Devel

**Single-machine**: The companion objects `boosted_trees.RegressionTree` and
`boosted_trees.GBRT` contain functions related to tree and forest model training
as well as using them for prediction. Some helper functions for
indexing are in `boosted_trees.Indexing` and miscellaneous ones
in `boosted_trees.Utils`. See example usage in various executable programs
in `boosted_trees` package.

**Distributed**: Similar to single machine, see respective companion objects
`boosted_trees.spark.SparkRegressionTree`, `boosted_trees.spark.SparkGBRT`
and `boosted_trees.spark.SparkIndexing`. For usage examples, see 
executable programs in `boosted_trees.spark` package.

