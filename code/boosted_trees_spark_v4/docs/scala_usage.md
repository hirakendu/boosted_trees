# Usage of Single-machine Scala Programs

These examples show the usage of single-machine Scala programs,
See the main [README](../README.md) for preliminaries and
[basic usage](../README.md#scala-programs-for-local-training).
Examples of detailed usage and utility programs is shown below.
Most options are self-explanatory with additional explanations
in the [Spark usage document](spark_usage.md).

## Detailed Usage

 1. **Decision tree model trainer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.DecisionTreeModelTrainer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/train_data.txt \
        --model-dir ${WORK}/tree \
        --loss-function square \
        --max-depth 5 \
        --min-gain-fraction 0.01 \
        --min-local-gain-fraction 1 \
        --min-count 2 \
        --min-weight 2.0 \
        --feature-weights-file "" \
        --use-sample-weights 0 \
        --histograms-method aggregate \
        --use-global-quantiles 1 \
        --fast-tree 1 \
        --batch-size 16 \
        --use-encoded-data 0 \
        --encoded-data-file ${WORK}/encoding/encoded_train_data.txt \
        --dicts-dir ${WORK}/encoding/dicts \
        --save-encoded-data 0

 2. **Decision tree error analyzer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.DecisionTreeErrorAnalyzer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/test_data.txt \
        --model-dir ${WORK}/tree \
        --error-dir ${WORK}/tree/error \
        --binary-mode 0 \
        --threshold 0.5

 3. **GBDT model trainer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.GBDTModelTrainer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/train_data.txt \
        --model-dir ${WORK}/forest \
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
        --histograms-method mapreduce \
        --use-global-quantiles 1 \
        --fast-tree 1 \
        --batch-size 16 \
        --use-encoded-data 0 \
        --encoded-data-file ${WORK}/encoding/encoded_train_data.txt \
        --dicts-dir ${WORK}/encoding/dicts \
        --save-encoded-data 0

 4. **GBDT error analyzer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.GBDTErrorAnalyzer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/test_data.txt \
        --model-dir ${WORK}/forest \
        --error-dir ${WORK}/forest/error \
        --binary-mode 0 \
        --threshold 0.5

## Utility Programs

 1. **Data sampler**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.DataSampler \
        --data-file ${WORK}/data.txt \
        --sample-data-file ${WORK}/sample_data.txt \
        --sample-fraction 0.01

 2. **Binary data generator**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.BinaryDataGenerator \
        --data-file ${WORK}/data.txt \
        --binary-data-file ${WORK}/binary_data.txt \
        --threshold 0.5

 3. **Train-test splitter**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.TrainTestSplitter \
        --data-file ${WORK}/data.txt \
        --train-data-file ${WORK}/split/train_data.txt \
        --test-data-file ${WORK}/split/test_data.txt \
        --train-fraction 0.8

 4. **Data encoder**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.DataEncoder \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/train_data.txt \
        --dicts-dir ${WORK}/encoding/dicts \
        --encoded-data-file ${WORK}/encoding/encoded_train_data.txt \
        --generate-dicts 1 \
        --encode-data 1

 5. **Weighted data generator**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.WeightedDataGenerator \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/train_data.txt \
        --weight-steps-file ${WORK}/weight_steps.txt \
        --weighted-data-header-file ${WORK}/weighted_data_header.txt \
        --weighted-data-file ${WORK}/weighted_train_data.txt

 6. **Feature weights generator**: generates a 0-1 feature weights file
    corresponding to features in header file and those specified in
    the includes and excludes files. The excludes are applied after includes.
    Use empty arguments (default) or don't specify the argument
    to include all features.  A better way to ignore features is by
    appending `#` character at the end of their names in the header file.

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.FeatureWeightsGenerator \
        --header-file ${WORK}/header.txt \
        --included-features-file ${WORK}/included_features.txt \
        --excluded-features-file ${WORK}/excluded_features.txt \
        --feature-weights-file ${WORK}/feature_weights.txt

 7. **Decision Tree Scorer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.DecisionTreeScorer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/test_data.txt \
        --model-dir ${WORK}/tree \
        --output-file ${WORK}/test_data_with_scores.txt \
        --binary-mode 0 \
        --threshold 0.5

 8. **GBDT Scorer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.GBDTScorer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/split/test_data.txt \
        --model-dir ${WORK}/forest \
        --output-file ${WORK}/test_data_with_scores.txt \
        --binary-mode 0 \
        --threshold 0.5

 9. **Classification Error Analyzer**:

        time \
        scala \
        -cp ${JARS}/boosted_trees_spark-spark.jar \
        boosted_trees.local.ClassificationErrorAnalyzer \
        --header-file ${WORK}/header.txt \
        --data-file ${WORK}/scores_labels.txt \
        --output-dir ${WORK}/error \
        --score-field score \
        --label-field label \
        --threshold 0.5
