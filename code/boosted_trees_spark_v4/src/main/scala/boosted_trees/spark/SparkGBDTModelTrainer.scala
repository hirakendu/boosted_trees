package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.scheduler.{SplitInfo, InputFormatInfo}
import org.apache.spark.deploy.SparkHadoopUtil

import boosted_trees.LabeledPoint

import boosted_trees.GBDTAlgorithmParameters
import boosted_trees.Node
import boosted_trees.loss._
import boosted_trees.GBDTModel


object SparkGBDTModelTrainer {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var headerFile: String = SparkDefaultParameters.headerFile
    var dataFile: String = SparkDefaultParameters.trainDataFile
    var modelDir: String = SparkDefaultParameters.forestModelDir
    var lossFunction: String = "square"
    var numTrees: Int = 5
    var shrinkage: Double = 0.8
    var maxDepth: Int = 4
    var minGainFraction: Double = 0.01
    var minLocalGainFraction: Double = 1
    var minCount: Int = 2
    var minWeight: Double = 2
    var minDistributedSamples: Int = 10000
    var featureWeightsFile: String = ""
    var useSampleWeights: Int = 0
    var useCache: Int = 1
    var histogramsMethod: String = "array-aggregate"
    var useGlobalQuantiles: Int = 1
    var fastTree: Int = 1
    var batchSize: Int = 16
    var persistInterval: Int = 1
    var numReducersPerNode: Int = 0
    var maxNumReducers: Int = 0
    var waitTimeMs: Long = 1
    var useEncodedData: Int = 0
    var encodedDataFile: String = SparkDefaultParameters.encodedTrainDataFile
    var dictsDir: String = SparkDefaultParameters.dictsDir
    var saveEncodedData: Int = 0

    var maxNumQuantiles: Int = 1000
    var maxNumQuantileSamples: Int = 100000
    var rngSeed: Long = 42

    // 0.1. Read parameters.

    var xargs: Array[String] = args
    if (args.length == 1) {
      xargs = args(0).split("\\^")
    }
    var argi: Int = 0
    while (argi < xargs.length) {
      if (("--header-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        headerFile = xargs(argi)
      } else if (("--data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        dataFile = xargs(argi)
      } else if (("--model-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        modelDir = xargs(argi)
      } else if (("--loss-function".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        lossFunction = xargs(argi)
      } else if (("--num-trees".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        numTrees = xargs(argi).toInt
      } else if (("--shrinkage".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        shrinkage = xargs(argi).toDouble
      } else if (("--max-depth".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxDepth = xargs(argi).toInt
      } else if (("--min-gain-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minGainFraction = xargs(argi).toDouble
      } else if (("--min-local-gain-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minLocalGainFraction = xargs(argi).toDouble
      } else if (("--min-count".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minCount = xargs(argi).toInt
      } else if (("--min-weight".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minWeight = xargs(argi).toDouble
      } else if (("--min-distributed-samples".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minDistributedSamples = xargs(argi).toInt
      } else if (("--feature-weights-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        featureWeightsFile = xargs(argi)
      } else if (("--use-sample-weights".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        useSampleWeights = xargs(argi).toInt
      } else if (("--use-cache".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        useCache = xargs(argi).toInt
      } else if (("--histograms-method".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        histogramsMethod = xargs(argi)
      } else if (("--use-global-quantiles".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        useGlobalQuantiles = xargs(argi).toInt
      } else if (("--fast-tree".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        fastTree = xargs(argi).toInt
      } else if (("--batch-size".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        batchSize = xargs(argi).toInt
      } else if (("--persist-interval".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        persistInterval = xargs(argi).toInt
      } else if (("--num-reducers-per-node".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        numReducersPerNode = xargs(argi).toInt
      } else if (("--max-num-reducers".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxNumReducers = xargs(argi).toInt
      } else if (("--wait-time-ms".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        waitTimeMs = xargs(argi).toLong
      } else if (("--use-encoded-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        useEncodedData = xargs(argi).toInt
      } else if (("--encoded-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        encodedDataFile = xargs(argi)
      } else if (("--dicts-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        dictsDir = xargs(argi)
      } else if (("--save-encoded-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        saveEncodedData = xargs(argi).toInt
      } else if (("--max-num-quantiles".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxNumQuantiles = xargs(argi).toInt
      } else if (("--max-num-quantile-samples".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxNumQuantileSamples = xargs(argi).toInt
      } else if (("--rng-seed".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        rngSeed = xargs(argi).toLong
      } else {
        println("\n  Error parsing argument \"" + xargs(argi) +
            "\".\n")
        return
      }
      argi += 1
    }

    // 0.2. Create Spark context.

    val localityCriticalFile: String =
        if (useEncodedData == 0) dataFile else encodedDataFile
    val preferredNodeLocationData: Map[String, Set[SplitInfo]] =
        InputFormatInfo.
        computePreferredLocations(Seq(new InputFormatInfo(SparkHadoopUtil.get.conf,
            classOf[org.apache.hadoop.mapred.TextInputFormat], localityCriticalFile))).
        map(entry => entry._1 -> entry._2.toSet).toMap

    val sc: SparkContext = new SparkContext((new SparkConf).setAppName(sparkAppName),
        preferredNodeLocationData)

    Thread.sleep(waitTimeMs)  // Sleep for specified time to ensure the workers are allocated.


    // 1. Read input data and index it.

    println("\n  Reading and indexing data.\n")

    // 1.1. Read header.

    val features: Array[String] = SparkUtils.readSmallFile(sc, headerFile).drop(1)
                    // .first.split("\t").drop(1)
    val numFeatures: Int = features.length
    val featureTypes: Array[Int] = features.map(feature => {
        if (feature.endsWith("$")) 1 else if (feature.endsWith("#")) -1 else 0})
        // 0 -> continuous, 1 -> discrete, -1 ignored
    var featureWeights: Array[Double] = Array.fill[Double](numFeatures)(1.0)
    if (!featureWeightsFile.equals("")) {
      featureWeights = SparkUtils.readSmallFile(sc, featureWeightsFile).drop(1).map(_.toDouble)
    }

    // 1.2 Read data and encode it.

    var indexes: Array[Map[String, Int]] = null
    var quantilesForFeatures: Array[Array[Double]] = null
    var samples: RDD[LabeledPoint] = null
    if (useEncodedData == 0) {
      val rawSamples: RDD[String] = sc.textFile(dataFile)
      // Index categorical features, find quantiles for continuous features
      // and encode data.
      indexes = SparkDataEncoding.generateIndexes(rawSamples, featureTypes)
      samples = SparkDataEncoding.encodeRawData(rawSamples, featureTypes, indexes)
      quantilesForFeatures = SparkDataEncoding.findQuantilesForFeatures(
          samples, featureTypes, maxNumQuantiles, maxNumQuantileSamples,
          rngSeed)

      // Save dicts and encoded data.
      if (saveEncodedData == 1) {
        SparkDataEncoding.saveIndexes(sc, dictsDir, features, indexes)
        SparkDataEncoding.saveQuantiles(sc, dictsDir, features, quantilesForFeatures)
        SparkDataEncoding.saveEncodedData(encodedDataFile, samples, featureTypes)
      }
    } else {
      // Use encoded data.
      samples = SparkDataEncoding.readEncodedData(sc, encodedDataFile)
      // Read indexes for categorical features.
      indexes = SparkDataEncoding.readIndexes(sc, dictsDir, features)
      // Read quantiles for continuous features.
      quantilesForFeatures = SparkDataEncoding.readQuantiles(sc, dictsDir, features)
    }
    val cardinalitiesForFeatures: Array[Int] = new Array(numFeatures)
    for (j <- 0 to numFeatures - 1) {
      if (featureTypes(j) == 1) {
        cardinalitiesForFeatures(j) = indexes(j).size
      }
    }


    // 2. Train forest model.

    println("\n  Training forest model.\n")

//    if (useCache == 1) {
//      samples.persist(StorageLevel.MEMORY_AND_DISK)
//      // samples.persist(StorageLevel.MEMORY_AND_DISK_2)
//      // samples.persist(StorageLevel.MEMORY_AND_DISK_SER)
//      // samples.persist
//      // samples.foreach(sample => {})  // Load now.
//    }

    val gbdtParams: GBDTAlgorithmParameters =
      GBDTAlgorithmParameters(
        featureTypes,
        numTrees,
        shrinkage,
        maxDepth,
        minGainFraction,
        minLocalGainFraction,
        minWeight,
        minCount,
        featureWeights,
        useSampleWeights,
        useCache,
        cardinalitiesForFeatures,
        useGlobalQuantiles,
        quantilesForFeatures,
        maxNumQuantiles,
        maxNumQuantileSamples,
        histogramsMethod,
        fastTree,
        batchSize,
        persistInterval,
        numReducersPerNode,
        maxNumReducers
      )

    var model: GBDTModel = null
    if (lossFunction.equals("entropy")) {
      val algorithm = new SparkGBDTAlgorithm[EntropyLossStats](
        new EntropyLoss, gbdtParams)
      model = algorithm.train(samples)
    } else {  // "square"
      val algorithm = new SparkGBDTAlgorithm[SquareLossStats](
        new SquareLoss, gbdtParams)
      model = algorithm.train(samples)
    }


    // 3. Print and save the forest.

    println("\n  Saving the forest.\n")

    model.loadNamesForIds(features, indexes)
    sc.parallelize(model.explain(), 1).saveAsTextFile(modelDir + "/forest.txt")
    sc.parallelize(model.save(), 1).saveAsTextFile(modelDir + "/forest.json")
    sc.parallelize(model.saveExtended(), 1).saveAsTextFile(modelDir + "/forest_ext.json")
    var linesForTrees = model.saveSimple()
    sc.parallelize(Array(numTrees), 1).saveAsTextFile(modelDir + "/forest_nodes/num_trees.txt")
    for (m <- 0 to numTrees - 1) {
      sc.parallelize(linesForTrees(m), 1).saveAsTextFile(modelDir + "/forest_nodes/nodes_" + m + ".txt")
    }

    linesForTrees = model.printDot()
    sc.parallelize(Array(numTrees), 1).saveAsTextFile(modelDir + "/forest_dot/num_trees.txt")
    for (m <- 0 to numTrees - 1) {
      sc.parallelize(linesForTrees(m), 1).saveAsTextFile(modelDir + "/forest_dot/tree_" + m + ".dot")
    }

    sc.parallelize(model.evaluateFeatureImportances().
        map(x => x._1 + "\t" + "%.2f".format(x._2)), 1).
        saveAsTextFile(modelDir + "/feature_importances.txt")

    sc.parallelize(model.evaluateFeatureSubsetImportances().
        map(x => x._1.mkString(",") + "\t" + "%.2f".format(x._2)), 1).
        saveAsTextFile(modelDir + "/feature_subset_importances.txt")

    sc.stop

  }

}
