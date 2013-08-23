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

The programs create and use the directory `/tmp/boosted_trees_work/`
by default for various working files and folders.
The dataset format consists of two parts:

 1. **Data file**: a plain text file containing one sample per line,
    each sample consisting of tab separated feature values.
    The first field/value in each line is considered to be the _label_.
    The values correspond to continuous (numbers) or
    categorical features (text).
    
 2. **Header/schema file:** a plain text file containing the feature names,
    one feature per line. The first line/feature name is always `label`.
    Feature names ending with the dollar character `$` are considered
    to be categorical.

By default, the expected data file is `/tmp/boosted_trees_work/data.txt`
and header file is `/tmp/boosted_trees_work/header.txt`.

A sample dataset is provided in `data/auto-mpg` folder along with
modeling results. This is a well known dataset taken from
[UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Auto+MPG).
Pre-processing steps are described in `data/auto-mpg/notes.txt`.
For an example run, copy this to the work folder as follows:

    rm -rf /tmp/boosted_trees_work && mkdir /tmp/boosted_trees_work
    cp -a data/auto-mpg/{data.txt,header.txt,split} /tmp/boosted_trees_work/

The program names and their parameters are mostly self-explanatory.
Sample commands for running each of these programs and their default
options are provided below with brief explanations.

 0. **Data sampler**: takes a small sample fraction of a big dataset.

        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.DataSampler \
          --data-file /tmp/boosted_trees_work/data.txt \
          --sample-data-file /tmp/boosted_trees_work/sample_data.txt \
          --sample-rate 0.01

 1. **Train-test splitter**: partitions the data into train and
 test sets: `/tmp/boosted_trees_work/split/train_data.txt` and
 `/tmp/boosted_trees_work/split/test_data.txt`.

        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.TrainTestSplitter \
          --data-file /tmp/boosted_trees_work/data.txt \
          --train-data-file /tmp/boosted_trees_work/split/train_data.txt \
          --test-data-file /tmp/boosted_trees_work/split/test_data.txt \
          --train-fraction 0.8
    
 2. **Data indexer**: indexes the categorical feature values observed
 in the train set to integer values and encodes the training data
 using the generated indexes. The indexes are used later
 on for prediction and test data analysis. For each categorical feature called
 `<feature_name>$`, it generates an index file
 `/tmp/boosted_trees_work/indexing/indexes/<feature_name>_index.txt`.
 The encoded train data is `/tmp/boosted_trees_work/indexing/indexed_train_data.txt`.

        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.DataIndexer \
          --header-file /tmp/boosted_trees_work/header.txt \
          --data-file /tmp/boosted_trees_work/split/train_data.txt \
          --indexes-dir /tmp/boosted_trees_work/indexing/indexes/ \
          --indexed-data-file /tmp/boosted_trees_work/indexing/indexed_train_data.txt \
          --save-indexed-data 1
  
 3. **Regression tree model trainer**: trains a regression tree model
 on training data. Training data can be raw or indexed -- in the former
 case, indexing is performed before training. The output model is saved to
 `/tmp/boosted_trees_work/indexing/tree/nodes.txt` which is used for prediction,
 for analyzing test error or runtime. An easy-to-read version is printed
 to `/tmp/boosted_trees_work/tree/tree.txt`.
 The baseline is trivial constant response, the average output of train data,
 and the corresponding RMSE (root of sample variance) and MAE. 
 
        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.RegressionTreeModelTrainer \
          --header-file /tmp/boosted_trees_work/header.txt \
          --data-file /tmp/boosted_trees_work/split/train_data.txt \
          --indexes-dir /tmp/boosted_trees_work/indexing/indexes/ \
          --indexed-data-file /tmp/boosted_trees_work/indexing/indexed_train_data.txt \
          --model-dir /tmp/boosted_trees_work/tree/ \
          --max-depth 5 \
          --min-gain-fraction 0.01 \
          --use-indexed-data 0 \
          --save-indexed-data 0
 
 4. **Regression tree error analyzer**: predicts on test data using a tree model
 and calculates test error in terms of RMSE (root mean square error)
 and MAE (mean absolute error). Test data can be raw or indexed -- in the former
 case, indexing is performed using same indexes as train data before prediction.
 Reads tree model from `/tmp/boosted_trees_work/tree/tree.txt`.
 The output errors are saved to `/tmp/boosted_trees_work/tree/error.txt`.
 
        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.RegressionTreeErrorAnalyzer \
          --header-file /tmp/boosted_trees_work/header.txt \
          --data-file /tmp/boosted_trees_work/split/test_data.txt \
          --indexes-dir /tmp/boosted_trees_work/indexing/indexes/ \
          --indexed-data-file /tmp/boosted_trees_work/indexing/indexed_test_data.txt \
          --model-dir /tmp/boosted_trees_work/tree/ \
          --error-file /tmp/boosted_trees_work/tree/error.txt \
          --use-indexed-data 0 \
          --save-indexed-data 0
      
 5. **Regression tree details printer**: prints detailed information
 about a tree model. Reads model from `/tmp/boosted_trees_work/tree/nodes.txt`
 and prints details to `/tmp/boosted_trees_work/tree/tree_details.txt`
 and `/tmp/boosted_trees_work/tree/nodes_details/`.
 
        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.RegressionTreeDetailsPrinter \
          --header-file /tmp/boosted_trees_work/header.txt \
          --indexes-dir /tmp/boosted_trees_work/indexing/indexes/ \
          --model-dir /tmp/boosted_trees_work/tree/

 7. **Regression tree DOT printer**: prints a Graphviz DOT file that
 illustrates a tree model. Also prints PDF if `graphviz` and `dot` are installed.
 Reads model from `/tmp/boosted_trees_work/tree/nodes.txt`
 and prints details to `/tmp/boosted_trees_work/tree/tree.dot`
 and `/tmp/boosted_trees_work/tree/tree/tree.pdf`.
 
        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.RegressionTreeDotPrinter \
          --header-file /tmp/boosted_trees_work/header.txt \
          --model-dir /tmp/boosted_trees_work/tree/
          
 8. **GBRT model trainer**: trains a GBRT forest model on training data.
 Most options are similar to regression tree model trainer.
 The output model is saved to `/tmp/boosted_trees_work/indexing/tree/forest/nodes/`.
 An easy-to-read version is printed to `/tmp/boosted_trees_work/forest/trees/`.
 
        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.GBRTModelTrainer \
          --header-file /tmp/boosted_trees_work/header.txt \
          --data-file /tmp/boosted_trees_work/split/train_data.txt \
          --indexes-dir /tmp/boosted_trees_work/indexing/indexes/ \
          --indexed-data-file /tmp/boosted_trees_work/indexing/indexed_train_data.txt \
          --model-dir /tmp/boosted_trees_work/forest/ \
          --num-trees 5 \
          --shrinkage 0.8 \
          --max-depth 4 \
          --min-gain-fraction 0.01 \
          --use-indexed-data 0 \
          --save-indexed-data 0
 
 9. **GBRT error analyzer**: predicts on test data using a forest model
 and calculates errors. Most options are similar to
 regression tree error analyzer.
 
        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.GBRTErrorAnalyzer \
          --header-file /tmp/boosted_trees_work/header.txt \
          --data-file /tmp/boosted_trees_work/split/test_data.txt \
          --indexes-dir /tmp/boosted_trees_work/indexing/indexes/ \
          --indexed-data-file /tmp/boosted_trees_work/indexing/indexed_test_data.txt \
          --model-dir /tmp/boosted_trees_work/forest/ \
          --error-file /tmp/boosted_trees_work/forest/error.txt \
          --use-indexed-data 0 \
          --save-indexed-data 0
      
 10. **GBRT details printer**: prints detailed information
 about a forest model.
 
        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.GBRTDetailsPrinter \
          --header-file /tmp/boosted_trees_work/header.txt \
          --indexes-dir /tmp/boosted_trees_work/indexing/indexes/ \
          --model-dir /tmp/boosted_trees_work/forest/
          
 11. **GBRT DOT printer**: prints Graphviz DOT files that
 illustrates the trees in a forest model.
 Also prints PDF if `graphviz` and `dot` are installed.
 
        time scala -cp target/boosted_trees_spark-spark.jar \
          boosted_trees.GBRTDotPrinter \
          --header-file /tmp/boosted_trees_work/header.txt \
          --model-dir /tmp/boosted_trees_work/forest/

### Distributed Spark programs

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
    SPARK_CLASSPATH=$(pwd)/target/boosted_trees_spark-spark.jar \
    spark_local_run.sh boosted_trees.spark.SparkDataSampler \
      --spark-master local[2,2] \
      --data-file file:///tmp/boosted_trees_work/data.txt \
      --sample-data-file file:///tmp/boosted_trees_work/sample_data.txt \
      --sample-rate 0.01 \
      2>spark_log.txt
      
or as caret separated:

    time \
    SPARK_CLASSPATH=$(pwd)/target/boosted_trees_spark-spark.jar \
    spark_local_run.sh boosted_trees.spark.SparkDataSampler \
    --spark-master^local[4]^\
    --data-file^file:///tmp/boosted_trees_work/data.txt^\
    --sample-data-file^file:///tmp/boosted_trees_work/sample_data.txt^\
    --sample-rate^0.01 \
      2>spark_log.txt

. To run on a Hadoop 2 (Yarn) cluster:

    time spark_yarn_run.sh \
      --jar $(pwd)/target/boosted_trees_spark-spark.jar \
      --class boosted_trees.spark.SparkDataSampler \
      --num-workers 4  \
      --worker-memory 3g \
      --worker-cores 2 \
      --queue default \
      --args \
    --spark-master^yarn-standalone^\
    --data-file^hdfs:///tmp/boosted_trees_work/data.txt^\
    --sample-data-file^hdfs:///tmp/boosted_trees_work/sample_data.txt^\
    --sample-rate^0.01

. For running on Yarn cluster, the `spark-master` is to be specified
as 	`yarn-standalone` and the arguments to the program are merged
into a single argument separated by carets `^` and passed as `args`.

The files and folders are to be specified with the protocol
and full path. Both file:// for local and hdfs:// for HDFS locations
are supported by Spark. Any text file written out by Spark using Hadoop
API is in the form of a folder with part files inside it.
Use the script `hadoop_txt_compact.sh` in the `scripts/` folder
to compact them into usual text files.

For an example run on the auto-mpg dataset on Yarn, copy it to HDFS:

    hadoop fs -rm -r -f /tmp/boosted_trees_work
    hadoop fs -mkdir -p /tmp/boosted_trees_work
    hadoop fs -copyFromLocal data/auto-mpg/data.txt data/auto-mpg/split data/auto-mpg/header.txt /tmp/boosted_trees_work/
    hadoop fs -ls /tmp/boosted_trees_work
.

Examples of the respective commands for distributed programs
using `spark_yarn_run.sh` similar to the single machine version
are provided below.

 1. **Spark Train-test splitter**:
 
        time spark_yarn_run.sh \
          --jar $(pwd)/target/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkTrainTestSplitter \
          --num-workers 4  \
          --worker-memory 2g \
          --worker-cores 2 \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --data-file^hdfs:///tmp/boosted_trees_work/data.txt^\
        --train-data-file^hdfs:///tmp/boosted_trees_work/split/train_data.txt^\
        --test-data-file^hdfs:///tmp/boosted_trees_work/split/test_data.txt^\
        --train-fraction^0.8
 .
     
 2. **Spark Data indexer**:

        time spark_yarn_run.sh \
          --jar $(pwd)/target/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkDataIndexer \
          --num-workers 4  \
          --worker-memory 2g \
          --worker-cores 2 \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^hdfs:///tmp/boosted_trees_work/header.txt^\
        --data-file^hdfs:///tmp/boosted_trees_work/split/train_data.txt^\
        --indexes-dir^hdfs:///tmp/boosted_trees_work/indexing/indexes/^\
        --indexed-data-file^hdfs:///tmp/boosted_trees_work/indexing/indexed_train_data.txt^\
        --save-indexed-data^1
 .
  
 3. **Spark Regression tree model trainer**:
 
        time spark_yarn_run.sh \
          --jar $(pwd)/target/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkRegressionTreeModelTrainer \
          --num-workers 4  \
          --worker-memory 2g \
          --worker-cores 2 \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^hdfs:///tmp/boosted_trees_work/header.txt^\
        --data-file^hdfs:///tmp/boosted_trees_work/split/train_data.txt^\
        --indexes-dir^hdfs:///tmp/boosted_trees_work/indexing/indexes/^\
        --indexed-data-file^hdfs:///tmp/boosted_trees_work/indexing/indexed_train_data.txt^\
        --model-dir^hdfs:///tmp/boosted_trees_work/tree/^\
        --max-depth^5^\
        --min-gain-fraction^0.01^\
        --min-distributed-samples^10000^\
        --use-indexed-data^0^\
        --save-indexed-data^0^\
        --cache-indexed-data^0
 .
 
 4. **Spark Regression tree error analyzer**:
 
        time spark_yarn_run.sh \
          --jar $(pwd)/target/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkRegressionTreeErrorAnalyzer \
          --num-workers 4  \
          --worker-memory 2g \
          --worker-cores 2 \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^hdfs:///tmp/boosted_trees_work/header.txt^\
        --data-file^hdfs:///tmp/boosted_trees_work/split/test_data.txt^\
        --indexes-dir^hdfs:///tmp/boosted_trees_work/indexing/indexes/^\
        --indexed-data-file^hdfs:///tmp/boosted_trees_work/indexing/indexed_test_data.txt^\
        --model-dir^hdfs:///tmp/boosted_trees_work/tree/^\
        --error-file^hdfs:///tmp/boosted_trees_work/tree/error.txt^\
        --use-indexed-data^0^\
        --save-indexed-data^0
 .
 
 5. **Spark GBRT model trainer**:
 
        time spark_yarn_run.sh \
          --jar $(pwd)/target/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkGBRTModelTrainer \
          --num-workers 4  \
          --worker-memory 2g \
          --worker-cores 2 \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^hdfs:///tmp/boosted_trees_work/header.txt^\
        --data-file^hdfs:///tmp/boosted_trees_work/split/train_data.txt^\
        --indexes-dir^hdfs:///tmp/boosted_trees_work/indexing/indexes/^\
        --indexed-data-file^hdfs:///tmp/boosted_trees_work/indexing/indexed_train_data.txt^\
        --residual-data-file^hdfs:///tmp/boosted_trees_work/indexing/residual_data.txt^\
        --model-dir^hdfs:///tmp/boosted_trees_work/tree/^\
        --num-trees^5^\
        --shrinkage^0.8^\
        --max-depth^4^\
        --min-gain-fraction^0.01^\
        --min-distributed-samples^10000^\
        --initial-num-trees^0^\
        --residual-mode^0^\
        --use-indexed-data^0^\
        --save-indexed-data^0^\
        --cache-indexed-data^0
 .
 
 6. **Spark GBRT error analyzer**:
 
        time spark_yarn_run.sh \
          --jar $(pwd)/target/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkGBRTErrorAnalyzer \
          --num-workers 4  \
          --worker-memory 2g \
          --worker-cores 2 \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^hdfs:///tmp/boosted_trees_work/header.txt^\
        --data-file^hdfs:///tmp/boosted_trees_work/split/test_data.txt^\
        --indexes-dir^hdfs:///tmp/boosted_trees_work/indexing/indexes/^\
        --indexed-data-file^hdfs:///tmp/boosted_trees_work/indexing/indexed_test_data.txt^\
        --model-dir^hdfs:///tmp/boosted_trees_work/forest/^\
        --error-file^hdfs:///tmp/boosted_trees_work/forest/error.txt^\
        --use-indexed-data^0^\
        --save-indexed-data^0
 .


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

