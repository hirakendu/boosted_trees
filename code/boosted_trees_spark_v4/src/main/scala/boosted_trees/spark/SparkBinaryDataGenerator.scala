package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object SparkBinaryDataGenerator {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var dataFile: String = SparkDefaultParameters.dataFile
    var binaryDataFile: String = SparkDefaultParameters.workDir + "/binary_data.txt"
    var threshold: Double = 0.5

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
      } else if (("--binary-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        binaryDataFile = xargs(argi)
      } else if (("--threshold".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        threshold = xargs(argi).toDouble
      } else {
        println("\n  Error parsing argument \"" + xargs(argi) +
            "\".\n")
        return
      }
      argi += 1
    }

    // 0.2. Create Spark context.

    val sc: SparkContext = new SparkContext((new SparkConf).setAppName(sparkAppName))


    // 1. Threshold data into binary.

    // 1.1. Read data.

    val samples: RDD[String] = sc.textFile(dataFile)

    // 1.2. Generate binary responses for samples.

    val binarySamples: RDD[String] = samples.map(sample => {
        val values: Array[String] =  sample.split("\t", -1)
        val response: Double = values(0).toDouble
        var b: Int = 0
        if (response >= threshold) {
          b = 1
        }
        b + "\t" + values.drop(1).mkString("\t")
      })

    // 1.3. Save binary data.
    binarySamples.saveAsTextFile(binaryDataFile)

    sc.stop

  }

}
