For an example run on the auto-mpg dataset on Yarn, copy it to HDFS:

    hadoop fs -rm -r -f temp/boosted_trees_work
    hadoop fs -mkdir -p temp/boosted_trees_work
    hadoop fs -copyFromLocal data/auto-mpg/data.txt data/auto-mpg/split data/auto-mpg/header.txt temp/boosted_trees_work/
    hadoop fs -ls temp/boosted_trees_work
.

Examples of the respective commands for distributed programs
using `spark_yarn_run.sh` similar to the single machine version
are provided below.

 1. **Data sampler**:

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
     
 2. **Binary data generator**:

        time spark_yarn_run.sh \
          --jar ${JARS}/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkBinaryDataGenerator \
          --num-workers 4 \
          --worker-memory 3g \
          --worker-cores 1 \
          --master-memory 3g \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --data-file^${DIST_WORK}/data.txt^\
        --binary-data-file^${DIST_WORK}/binary_data.txt^\
        --threshold^0.5
            
 3. **Train-test splitter**:

        time spark_yarn_run.sh \
          --jar ${JARS}/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkTrainTestSplitter \
          --num-workers 4 \
          --worker-memory 3g \
          --worker-cores 1 \
          --master-memory 3g \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --data-file^${DIST_WORK}/data.txt^\
        --train-data-file^${DIST_WORK}/split/train_data.txt^\
        --test-data-file^${DIST_WORK}/split/test_data.txt^\
        --train-fraction^0.8
     
 4. **Data indexer**:

        time spark_yarn_run.sh \
          --jar ${JARS}/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkDataIndexer \
          --num-workers 4 \
          --worker-memory 3g \
          --worker-cores 1 \
          --master-memory 3g \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^${DIST_WORK}/header.txt^\
        --data-file^${DIST_WORK}/split/train_data.txt^\
        --indexes-dir^${DIST_WORK}/indexing/indexes/^\
        --indexed-data-file^${DIST_WORK}/indexing/indexed_train_data.txt^\
        --save-indexed-data^1
     
 5. **Weighted data generator**:

        time spark_yarn_run.sh \
          --jar ${JARS}/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkWeightedDataGenerator \
          --num-workers 4 \
          --worker-memory 3g \
          --worker-cores 1 \
          --master-memory 3g \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^${DIST_WORK}/header.txt^\
        --weighted-data-header-file^${DIST_WORK}/weighted_data_header.txt^\
        --data-file^${DIST_WORK}/indexing/indexed_train_data.txt^\
        --weighted-data-file^${DIST_WORK}/indexing/indexed_weighted_train_data.txt
     
 6. **Regression tree model trainer**:

        time spark_yarn_run.sh \
          --jar ${JARS}/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkRegressionTreeModelTrainer \
          --num-workers 4 \
          --worker-memory 3g \
          --worker-cores 1 \
          --master-memory 3g \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
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
        --cache-indexed-data^0
     
 7. **Regression tree error analyzer**:

        time spark_yarn_run.sh \
          --jar ${JARS}/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkRegressionTreeErrorAnalyzer \
          --num-workers 4 \
          --worker-memory 3g \
          --worker-cores 1 \
          --master-memory 3g \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^${DIST_WORK}/header.txt^\
        --data-file^${DIST_WORK}/split/test_data.txt^\
        --indexes-dir^${DIST_WORK}/indexing/indexes/^\
        --indexed-data-file^${DIST_WORK}/indexing/indexed_test_data.txt^\
        --model-dir^${DIST_WORK}/tree/^\
        --error-file^${DIST_WORK}/tree/error.txt^\
        --roc-file^${DIST_WORK}/forest/roc.txt^\
        --binary-mode^0^\
        --threshold^0.5^\
        --max-num-roc-samples^100000^\
        --use-indexed-data^0^\
        --save-indexed-data^0
     
 8. **GBRT model trainer**:

        time spark_yarn_run.sh \
          --jar ${JARS}/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkGBRTModelTrainer \
          --num-workers 4 \
          --worker-memory 3g \
          --worker-cores 1 \
          --master-memory 3g \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
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
        --cache-indexed-data^0
     
 9. **GBRT error analyzer**:

        time spark_yarn_run.sh \
          --jar ${JARS}/boosted_trees_spark-spark.jar \
          --class boosted_trees.spark.SparkGBRTErrorAnalyzer \
          --num-workers 4 \
          --worker-memory 3g \
          --worker-cores 1 \
          --master-memory 3g \
          --queue default \
          --args \
        --spark-master^yarn-standalone^\
        --header-file^${DIST_WORK}/header.txt^\
        --data-file^${DIST_WORK}/split/test_data.txt^\
        --indexes-dir^${DIST_WORK}/indexing/indexes/^\
        --indexed-data-file^${DIST_WORK}/indexing/indexed_test_data.txt^\
        --model-dir^${DIST_WORK}/forest/^\
        --error-file^${DIST_WORK}/forest/error.txt^\
        --roc-file^${DIST_WORK}/forest/roc.txt^\
        --binary-mode^0^\
        --threshold^0.5^\
        --max-num-roc-samples^100000^\
        --use-indexed-data^0^\
        --save-indexed-data^0
