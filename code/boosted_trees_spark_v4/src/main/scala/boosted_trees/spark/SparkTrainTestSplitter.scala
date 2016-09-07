package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object SparkTrainTestSplitter {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var dataFile: String = SparkDefaultParameters.dataFile
    var trainDataFile: String = SparkDefaultParameters.trainDataFile
    var testDataFile: String = SparkDefaultParameters.testDataFile
    var trainFraction: Double = 0.8
    var rngSeed: Long = 42

    // 0.1. Read parameters.

    var xargs: Array[String] = args
    if (args.length == 1) {
      xargs = args(0).split("\\^")
    }
    var argi: Int = 0
    while (argi < xargs.length) {
      if (("--data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        dataFile = xargs(argi)
      } else if (("--train-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        trainDataFile = xargs(argi)
      } else if (("--test-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        testDataFile = xargs(argi)
      } else if (("--train-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        trainFraction = xargs(argi).toDouble
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

    val sc: SparkContext = new SparkContext((new SparkConf).setAppName(sparkAppName))


    // 1. Generate train and test samples.

    val rawSamples: RDD[String] = sc.textFile(dataFile)
    val trainRawSamples: RDD[String] = rawSamples.sample(false, trainFraction, rngSeed)
    val testRawSamples: RDD[String] = rawSamples.subtract(trainRawSamples)


    // 2. Save.

    trainRawSamples.saveAsTextFile(trainDataFile)
    testRawSamples.saveAsTextFile(testDataFile)

    sc.stop

  }

}
