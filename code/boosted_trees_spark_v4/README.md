# Boosted Trees on Spark and Scala

This package provides libraries and applications for
distributed training of boosted tree models using Spark and Scala.
Salient functionality includes:

 * Decision trees and gradient boosting.
 * Regression using square loss, binary classification using entropy loss,
   and generic algorithm implementations using interfaces that
   allow for other loss functions.
 * Support for both continuous and categorical features.
 * Feature weights and sample weights.
 * Scalable implementations using Spark for large distributed datasets,
   as well as fast single-machine implementations using only Scala
   (without Spark) for medium-sized local datasets.

## Installation and Setup

To get started, simply use the pre-compiled
[jar](target/boosted_trees_spark-spark.jar)
in the [target](target/) folder as a Spark application jar,
meant to be used with Spark 0.9.

To build from source, run:

    mvn package

in this Maven project folder to generate the jar files in the `target/` folder.
Assumes Java and Maven are installed. Pass the appropriate versions
of Spark, Scala and Hadoop as Java properties if required.
E.g., to compile for Spark 0.8, Scala 2.9 and Hadoop 0.23, run

    mvn package \
      -Dspark.version=0.8.1-incubating \
      -Dscala.binary.version=2.9.3 \
      -Dscala.version=2.9.3 \
      -Dhadoop.version=0.23.10

.

## Data Format

The data is expected to be in tab-delimited format along with
a separate header file for the model training and testing programs.
Thus, a dataset consists of two parts:

 1. **Data file**: a plain text file containing one sample per line,
    each sample consisting of tab separated feature values.
    The first value in each line is considered to be the `label`,
    and has to be numerical, in particular, 0/1 for binary classification.
    The other values can be continuous (numbers) or
    categorical features (text).

 2. **Header/schema file:** a plain text file containing the feature names,
    one feature per line. The first line/feature name corresponds to the `label`.
    Feature names ending with the dollar character `$` are considered
    to be categorical. Beyond these rules, the actual names don't matter.
    A convenient way to ignore features is by appending the hash character `#`
    at the end.

As an example, see the files
[data.txt](data/auto-mpg/data.txt) and [header.txt](data/auto-mpg/header.txt)
in [data/auto-mpg/](data/auto-mpg/) folder.

## Spark Programs for Distributed Training

The following examples show basic usage of the Spark programs
for distributed training. Detailed usage examples and utility programs
are shown in [docs/spark_usage.md](docs/spark_usage.md).
Details and examples for running on YARN cluster are shown in
[docs/spark_yarn_usage.md](docs/spark_yarn_usage.md).

In the examples, it is assumed that the working directory is `DIST_WORK`,
which may be a location in local file-system or HDFS, e.g.,

    export DIST_WORK="file:///${HOME}/temp/boosted_trees_work/"  # for local
    # or
    export DIST_WORK="hdfs:///user/${USER}/temp/boosted_trees_work/"  # for cluster

. A working directory on local filesystem to fetch the results and analyze
further is assumed to be `WORK`, e.g., `WORK=${HOME}/temp/boosted_trees_work`.
It may be different from `DIST_WORK`, or alternatively, set
`DIST_WORK=file://${WORK}` if running Spark in local mode.
The spark application jar for boosted trees is assumed to be located
in local filesystem at `JARS` location, e.g., `JARS=${HOME}/opt/jars`.
Although the examples use `local[*]` as Spark master,
i.e., `SPARK_MASTER=local[*]`, a single-machine emulation of Spark cluster
with all cpu cores of the machine,
the usage is similar for other cluster settings.
Lastly it is assumed that the Spark installation bin containing
`spark-submit` launcher script is in path, i.e.,
`PATH=${PATH}:${SPARK_HOME}/bin`.

 1. **Decision tree model trainer**: trains a
    regression tree model using the specified data and header file,
    and saves it to the specified model directory.
    The basic model parameters that are used as stopping criteria
    for tree-growing process are the maximum depth of the tree
    and the minimum gain a split should have compared to the
    sample variance of the full dataset.
    The variety of modeling and performance parameters,
    including loss functions, feature and sample weights,
    caching and learning strategies are covered in the full documentation.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkDecisionTreeModelTrainer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --model-dir ${DIST_WORK}/tree \
        --max-depth 5 \
        --min-gain-fraction 0.01 \
        2>/tmp/spark_log.txt

    The model directory contains JSON representations of the trained tree model
    that can be used for prediction, a Graphviz DOT file that can
    be used for PDF illustrations (use `dot -T pdf tree.dot -o tree.pdf`),
    and feature importances. Note that the output files are in the form
    of part files and need to be _compacted_. Checking the output:

        cd $WORK
        # hadoop fs -copyToLocal $DIST_WORK/tree .
        less tree/tree.json/part-00000
        less tree/feature_importances.txt/part-00000
        dot -T pdf tree/tree.dot/part-00000 -o tree/tree.pdf  # Requires Graphviz.

 2. **Decision tree error analyzer**: predicts on test data using a given
    tree model and outputs test error in terms of
    RMSE (root mean square error), MAE (mean absolute error)
    and scatter plots. Additional error statistics like TPR, FPR, AUC
    are calculated for binary classification.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkDecisionTreeErrorAnalyzer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/tree \
        --error-dir ${DIST_WORK}/tree/error \
        2>/tmp/spark_log.txt
        
        cd $WORK
        # hadoop fs -copyToLocal $DIST_WORK/tree .
        less tree/error/error.txt/part-00000

 3. **GBDT model trainer**: trains a GBDT forest model.
    Most parameters are similar to regression tree model trainer.
    Apart from the modeling parameters for decision tree model
    trainer, that are applicable to individual trees,
    the main parameters are the number of trees and shrinkage factor.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkGBDTModelTrainer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/train_data.txt \
        --model-dir ${DIST_WORK}/forest \
        --num-trees 5 \
        --shrinkage 0.8 \
        --max-depth 4 \
        --min-gain-fraction 0.01 \
        2>/tmp/spark_log.txt
        
        cd $WORK
        # hadoop fs -copyToLocal $DIST_WORK/forest .
        less forest/forest.json/part-00000
        less forest/feature_importances.txt/part-00000

 4. **GBDT error analyzer**: predicts on test data using a given
    forest model and outputs test error, similar to decision tree error analyzer.

        time \
        spark-submit \
        --master ${SPARK_MASTER} \
        --class boosted_trees.spark.SparkGBDTErrorAnalyzer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/forest \
        --error-dir ${DIST_WORK}/forest/error \
        2>/tmp/spark_log.txt
        
        cd $WORK
        # hadoop fs -copyToLocal $DIST_WORK/forest .
        less forest/error/error.txt/part-00000

## Scala Programs for Local Training

These examples show the basic usage of single-machine Scala programs,
suitable for fast training on datasets of size < 1 GB and with less
than 100 million samples. Detailed usage examples and utility programs
are shown in [docs/scala_usage.md](docs/scala_usage.md).
They are analogous to the Spark programs.

In the examples, it is assumed that the working directory is `WORK`,
a location in local file-system, e.g., `WORK=${HOME}/temp/boosted_trees_work`.
The boosted trees application jar is assumed to be located
in local filesystem at `JARS` location, e.g., `JARS=${HOME}/opt/jars`.
It is assumed that the Scala installation bin containing
`scala` binary is in `PATH`, i.e., `PATH=${PATH}:${SCALA_HOME}/bin`.

 1. **Decision tree model trainer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.DecisionTreeModelTrainer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/train_data.txt \
        --model-dir ${WORK}/tree \
        --max-depth 5 \
        --min-gain-fraction 0.01
        
        cd $WORK
        less tree/tree.json
        less tree/feature_importances.txt
        dot -T pdf tree/tree.dot -o tree/tree.pdf

 2. **Decision tree error analyzer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.DecisionTreeErrorAnalyzer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/test_data.txt \
        --model-dir ${WORK}/tree \
        --error-dir ${WORK}/tree/error
        
        cd $WORK
        less tree/error/error.txt

 3. **GBDT model trainer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.GBDTModelTrainer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/train_data.txt \
        --model-dir ${WORK}/forest \
        --num-trees 5 \
        --shrinkage 0.8 \
        --max-depth 4 \
        --min-gain-fraction 0.01
        
        cd $WORK
        less forest/forest.json
        less forest/feature_importances.txt

 4. **GBDT error analyzer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.GBDTErrorAnalyzer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/test_data.txt \
        --model-dir ${WORK}/forest \
        --error-dir ${WORK}/forest/error
        
        cd $WORK
        less forest/error/error.txt
