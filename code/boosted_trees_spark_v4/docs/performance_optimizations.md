# Optimizing Spark Configuration and GBDT Parameters for Performance

## General optimization tips

 1. Use 64-bit JDK and use `7500m` executor memory and driver memory.
    Higher is typically better, depending on the (YARN) cluster limits
    on container memory.
    
 2. Compression schemes like bzip2 take a lot of CPU and memory
    and cause frequent GC, so in general avoid compression with Spark
    when executors/containers are memory limited.
    
 3. Spark performance (and Hadoop in general) is sensitive to input task sizes.
    Larger input task sizes are better, as long as they fit in memory.
    Using `spark.storage.memoryFraction=0.3` with executor memory `7500m`
    gives about 2 GB per executor for caching input.
    Note that 1 GB of uncompressed text typically takes about 1.5 GB in
    Java object format in memory, so it should fit in a single executor.
    Too large a task size may not fit in a single executor's memory.
    In that case, dataset may not be cached in memory, and use disk instead.
    On memory constrained executors, input task sizes larger than 512 MB
    may be unstable due to frequent GC. So input task sizes of 256 MB or 512 MB,
    e.g., `spark.hadoop.mapred.min.split.size=$((2*128*1024*1024))` and
    `spark.hadoop.mapred.max.split.size=$((2*128*1024*1024))`,
    are a good tradeoff between performance and stability.
    
    Each part-file of input data is split into specified input task sizes.
    If the part-files are smaller, each one is a separate task.
    Too many tasks may cause some scheduling and shuffle overhead.
    Hence, if there are too many small part files in the input data,
    a Hadoop streaming job may be used to uncompress and/or re-partition
    the input data appropriately into sufficiently large part files,
    typically around 1 to 2 GB each. The following is one such
    Hadoop-streaming command that can be used for uniformly re-parititioning
    and compressing/decompressing data:
    
        time \
        hadoop jar ${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming.jar \
        -D mapred.job.queue.name=${QUEUE} \
        -D mapred.output.compress=false \
        -D mapred.output.compression.codec=org.apache.hadoop.io.compress.BZip2Codec \
        -input ${INPUT} \
        -output ${OUTPUT} \
        -mapper "cat -n" \
        -reducer "cut -f 2-" \
        -numReduceTasks ${NUM_OUTPUT_PARTS}
    
    The `QUEUE`, `INPUT`, `OUTPUT` and `NUM_OUTPUT_PARTS` need to specified appropriately.
    The line number as the key for uniform key distribution.
    Preferably use a prime number for `NUM_OUTPUT_PARTS`, e.g., see
    [http://primes.utm.edu/lists/small/100000.txt](http://primes.utm.edu/lists/small/100000.txt).
    Use `compress=true` for compression and `GzipCodec` (small `z`, unlike in `Bzip2Codec`)
    for gzip compression.
    
    Similar to issues with compressed input in Spark programs on memory constrained executors,
    it is better to avoid output compression in Spark programs as well.
    Instead, a similar post-processing job as above may be used for compression
    and re-partitioning into smaller number of parts, which can help
    in saving both storage space and namespace quota in HDFS.
    
 4. The GBDT model training code performs an indexing of the categorical features
    and encoding the data using the indexes/dictionaries, before the actual model training.
    A separate data encoder program is provided to perform this step separately
    before actual model training. It helps in 2 ways:
    1. Since it mostly involves map operations, many executors with less memory
       can be used to speed up this step.
    2. Repartition the resulting encoded data if needed.
    
 5. Some additional Spark configuration settings can be used
    for improved scheduling and speculative execution.
    These are mentioned in
    [yspark_yarn twiki](http://twiki.corp.yahoo.com/view/Grid/SparkOnYarnProduct),
    as well as upstream Spark documentation for
    [submitting applications](http://spark.apache.org/docs/latest/submitting-applications.html),
    and [tuning](http://spark.apache.org/docs/latest/tuning.html).
    Some of these settings have been put together in the form of wrapper scripts,
    [spark bundle](https://git.corp.yahoo.com/hdas/spark_bundle_1.0),
    around the `spark-submit` script provided by upstream Spark.

## Example optimized modeling

 1. Run these two commands to use the Spark installation and GBDT jars
    in `/homes/hdas/pub/opt` on Yahoo Grid:
    
        source /homes/hdas/pub/opt/spark_launcher_scripts_1.0/spark_ygrid_env_pub.sh
        export JARS=/homes/hdas/pub/opt/jars
    
    This will provide a script called `spark_run.sh` which is a thin wrapper
    around `spark-submit` with additional configuration and can be used
    as a drop-in replacement for `spark-submit`.
    The `$JARS` folder contains a latest version of GBDT jar file,
    should be same as that in git repo.

    The actual configuration script for reference is
    [spark_bundle/scripts/spark_launcher_scripts/spark_env.sh](https://git.corp.yahoo.com/hdas/spark_bundle_1.0/blob/master/scripts/spark_launcher_scripts_1.0/spark_env.sh).
    
    Thereafter, the Spark GBDT programs can be run as given in
    [spark yarn usage](spark_yarn_usage.md), but with `spark_run.sh`
    instead of `spark-submit` and with added options as required.
    
 2. Run data encoding: As in the Spark GBDT documentation,
    the following commands assume the inputs to be `${DIST_WORK}/header.txt`
    and `${DIST_WORK}/split/train_data.txt`.
    Select number of executors and memory appropriately.
    If training data is compressed, use `7500m` executor memory,
    task split size of 128 MB.
    If it's plain text, use task split size of 512 MB and say `2000m`
    memory per executor. It's always safe to use the largest amount of
    driver memory, say `7500m`. Set number of executors appropriately,
    say 1 GB of uncompressed data per executor.
    For example, if the data is 50 GB uncompressed:
    
        export QUEUE="default"
        export NUM_EXECUTORS=100
        export EXECUTOR_MEMORY=3000m
        export DRIVER_MEMORY=3000m
        
        time \
        SPARK_SUBMIT_OPTS="${SPARK_SUBMIT_OPTS} \
          -Dspark.ui.acls.enable=false \
          -Dspark.ui.view.acls=* \
          -Dspark.hadoop.mapred.min.split.size=$((1*128*1024*1024)) \
          -Dspark.hadoop.mapred.max.split.size=$((1*128*1024*1024)) \
          -Dspark.shuffle.memoryFraction=0.05 \
          -Dspark.storage.memoryFraction=0.05" \
        spark_run.sh \
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
        --max-num-quantiles 100 \
        --max-num-quantile-samples 100000 \
        --wait-time-ms 15000
    
    Note that the above example uses 100 quantile bins for each continuous feature.
    
    Re-partition the encoded data if necessary into roughly 1 GB part files.
    Only do this if the part files are smaller than say 64 MB.
    
        hadoop fs -du -s -h ${DIST_WORK}/encoding/encoded_train_data.txt/*
        hadoop fs -mv ${DIST_WORK}/encoding/encoded_train_data.txt ${DIST_WORK}/encoding/encoded_train_data_many.txt
        
        time \
        hadoop jar ${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming.jar \
        -D mapred.job.queue.name=${QUEUE} \
        -D mapred.output.compress=true \
        -D mapred.output.compression.codec=org.apache.hadoop.io.compress.BZip2Codec \
        -input ${DIST_WORK}/encoding/encoded_train_data_many.txt \
        -output ${DIST_WORK}/encoding/encoded_train_data.txt \
        -mapper "cat -n" \
        -reducer "cut -f 2-" \
        -numReduceTasks 47
        
        hadoop fs -du -s -h ${DIST_WORK}/encoding/encoded_train_data*.txt
        hadoop fs -rm -r -f -skipTrash ${DIST_WORK}/encoding/encoded_train_data_many.txt
    
 3. Run model training:
    
        export QUEUE="APG"
        export NUM_EXECUTORS=50
        export EXECUTOR_MEMORY=7500m
        export DRIVER_MEMORY=7500m

        time \
        SPARK_SUBMIT_OPTS="${SPARK_SUBMIT_OPTS} \
          -Dspark.hadoop.mapred.min.split.size=$((4*128*1024*1024)) \
          -Dspark.hadoop.mapred.max.split.size=$((4*128*1024*1024)) \
          -Dspark.shuffle.memoryFraction=0.05 \
          -Dspark.storage.memoryFraction=0.4 \
          -Dspark.speculation=true \
          -Dspark.akka.frameSize=1024 \
          -Dspark.locality.wait.process=4000 -Dspark.locality.wait.node=6000 -Dspark.locality.wait.rack=16000 \
          -Dspark.scheduler.executorTaskBlacklistTime=800 -Dspark.task.maxFailures=8 -Dspark.yarn.max.worker.failures=10000 \
          " \
        spark_run.sh \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkGBDTModelTrainer \
        ${JARS}/boosted_trees_spark-spark.jar \
        --header-file ${DIST_WORK}/gbdt_header.txt \
        --use-encoded-data 1 \
        --encoded-data-file ${DIST_WORK}/encoding/encoded_train_data.txt \
        --dicts-dir ${DIST_WORK}/encoding/dicts \
        --model-dir ${DIST_WORK}/cls_forest \
        --loss-function entropy \
        --num-trees 35 \
        --shrinkage 0.1 \
        --max-depth 6 \
        --min-gain-fraction 1e-5 \
        --min-local-gain-fraction 1 \
        --use-cache 1 \
        --histograms-method array-aggregate \
        --fast-tree 1 \
        --num-reducers-per-node 8 \
        --max-num-reducers 64 \
        --batch-size 64 \
        --persist-interval 10 \
        --wait-time-ms 30000
    
    If the dataset has mostly continuous features and/or categorical features with small cardinality, use the histograms-method array-aggregate. It's typically faster than aggregate. See documentation for details. Use a batch-size of 64 to begin with, but if for deeper levels, the model training runs slow or fails due to out-of-memory, limit the batch size to 2^d, where d is the last depth it runs fast. If there are several categorical features with high cardinality > 10,000, e.g., advertiser campaign ids or property spaceids or browser type and version etc, it may be better to use aggregate.
    
 4. Model evaluation.
    
    Mostly straight-forward use of `SparkGBDTErrorAnalyzer`.
    For binary classification, for AUC calculations
    on full dataset, i.e.,  `full-auc = 1`, better to have
    `use-cache = 1` and `spark.storage.memoryFraction=0.4`.
    Note that test data should not be encoded using indexes.
    Also, indexing is internal to the model training algorithm and
    there are no references or dependencies on the indexing
    in the generated model.
    
        time \
        SPARK_SUBMIT_OPTS="${SPARK_SUBMIT_OPTS} \
          -Dspark.ui.acls.enable=false \
          -Dspark.ui.view.acls=* \
          -Dspark.hadoop.mapred.min.split.size=$((2*128*1024*1024)) \
          -Dspark.hadoop.mapred.max.split.size=$((2*128*1024*1024)) \
          -Dspark.storage.memoryFraction=0.3" \
        spark_run.sh \
        --master yarn-cluster \
        --queue ${QUEUE} \
        --num-executors ${NUM_EXECUTORS} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${DRIVER_MEMORY} \
        --class boosted_trees.spark.SparkGBDTErrorAnalyzer \
        ${JARS}/boosted_trees_spark_v4-spark.jar \
        --header-file ${DIST_WORK}/gbdt_header.txt \
        --data-file ${DIST_WORK}/split/test_data.txt \
        --model-dir ${DIST_WORK}/cls_forest \
        --error-dir ${DIST_WORK}/cls_forest/error \
        --binary-mode 1 \
        --threshold 0.5 \
        --full-auc 1 \
        --max-num-summary-samples 100000 \
        --use-cache 1 \
        --wait-time-ms 15000
