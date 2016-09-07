package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.scheduler.{SplitInfo, InputFormatInfo}
import org.apache.spark.deploy.SparkHadoopUtil

import boosted_trees.LabeledPoint


object SparkDataEncoder {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var headerFile: String = SparkDefaultParameters.headerFile
    var dataFile: String = SparkDefaultParameters.trainDataFile
    var dictsDir: String = SparkDefaultParameters.dictsDir
    var encodedDataFile: String = SparkDefaultParameters.encodedTrainDataFile
    var generateDicts: Int = 1
    var encodeData: Int = 1
    var waitTimeMs: Long = 1

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
      } else if (("--dicts-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        dictsDir = xargs(argi)
      } else if (("--encoded-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        encodedDataFile = xargs(argi)
      } else if (("--generate-dicts".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        generateDicts = xargs(argi).toInt
      } else if (("--encode-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        encodeData = xargs(argi).toInt
      } else if (("--max-num-quantiles".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxNumQuantiles = xargs(argi).toInt
      } else if (("--max-num-quantile-samples".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxNumQuantileSamples = xargs(argi).toInt
      } else if (("--rng-seed".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        rngSeed = xargs(argi).toLong
      } else if (("--wait-time-ms".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        waitTimeMs = xargs(argi).toLong
      } else {
        println("\n  Error parsing argument \"" + xargs(argi) +
            "\".\n")
        return
      }
      argi += 1
    }

    // 0.2. Create Spark context.

    val localityCriticalFile: String = dataFile
    val preferredNodeLocationData: Map[String, Set[SplitInfo]] =
        InputFormatInfo.
        computePreferredLocations(Seq(new InputFormatInfo(SparkHadoopUtil.get.conf,
            classOf[org.apache.hadoop.mapred.TextInputFormat], localityCriticalFile))).
        map(entry => entry._1 -> entry._2.toSet).toMap

    val sc: SparkContext = new SparkContext((new SparkConf).setAppName(sparkAppName),
        preferredNodeLocationData)

    Thread.sleep(waitTimeMs)  // Sleep for specified time to ensure the workers are allocated.


    // 1. Read input data and index it.

    // 1.1. Read header.

    val features: Array[String] = SparkUtils.readSmallFile(sc, headerFile).drop(1)
                    // .first.split("\t").drop(1)
    val numFeatures: Int = features.length
    val featureTypes: Array[Int] = features.map(feature => {
        if (feature.endsWith("$")) 1 else if (feature.endsWith("#")) -1 else 0})
        // 0 -> continuous, 1 -> discrete, -1 ignored

    // 1.2 Read data.

    val rawSamples: RDD[String] = sc.textFile(dataFile)

    // 1.3. Index categorical features, find quantiles for continuous features
    //      and encode data.

    println("\n  Generating/reading indexes.\n")
    var indexes:  Array[Map[String,Int]] = null
    var quantilesForFeatures: Array[Array[Double]] = null
    var samples: RDD[LabeledPoint] = null
    if (generateDicts == 1) {
      indexes = SparkDataEncoding.generateIndexes(rawSamples, featureTypes)
      samples = SparkDataEncoding.encodeRawData(rawSamples, featureTypes, indexes)
      quantilesForFeatures = SparkDataEncoding.findQuantilesForFeatures(
          samples, featureTypes, maxNumQuantiles, maxNumQuantileSamples, rngSeed)
      SparkDataEncoding.saveIndexes(sc, dictsDir, features, indexes)
      SparkDataEncoding.saveQuantiles(sc, dictsDir, features, quantilesForFeatures)
    } else {
      indexes = SparkDataEncoding.readIndexes(sc, dictsDir, features)
      samples = SparkDataEncoding.encodeRawData(rawSamples, featureTypes, indexes)
        // Lazy materialization.
    }

    // 1.4. Save encoded data.

    if (encodeData == 1) {
      println("\n  Saving encoded data.\n")
      SparkDataEncoding.saveEncodedData(encodedDataFile, samples, featureTypes)
    }

    sc.stop

  }

}
