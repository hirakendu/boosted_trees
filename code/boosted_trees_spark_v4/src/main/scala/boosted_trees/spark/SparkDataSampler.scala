package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object SparkDataSampler {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var dataFile: String = SparkDefaultParameters.dataFile
    var sampleDataFile: String = SparkDefaultParameters.workDir + "/sample_data.txt"
    var sampleFraction: Double = 0.01
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
      } else if (("--sample-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        sampleDataFile = xargs(argi)
      } else if (("--sample-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        sampleFraction = xargs(argi).toDouble
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


    // 1. Take samples and save.

    sc.textFile(dataFile).sample(false, sampleFraction, rngSeed).saveAsTextFile(sampleDataFile)

    sc.stop

  }

}
