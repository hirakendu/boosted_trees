package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object SparkWeightedDataGenerator {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var headerFile: String = SparkDefaultParameters.headerFile
    var dataFile: String = SparkDefaultParameters.trainDataFile
    var weightStepsFile: String = SparkDefaultParameters.workDir + "/weight_steps.txt"
    var weightedDataHeaderFile: String = SparkDefaultParameters.workDir + "/weighted_data_header.txt"
    var weightedDataFile: String = SparkDefaultParameters.workDir + "/weighted_train_data.txt"

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
      } else if (("--weight-steps-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        weightStepsFile = xargs(argi)
      } else if (("--weighted-data-header-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        weightedDataHeaderFile = xargs(argi)
      }   else if (("--weighted-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        weightedDataFile = xargs(argi)
      } else {
        println("\n  Error parsing argument \"" + xargs(argi) +
            "\".\n")
        return
      }
      argi += 1
    }

    // 0.2. Create Spark context.

    val sc: SparkContext = new SparkContext((new SparkConf).setAppName(sparkAppName))


    // 1. Read data and add weight column.

    println("\n  Reading data and adding weight column.\n")

    // 1.1. Read header.

    val features: Array[String] = SparkUtils.readSmallFile(sc, headerFile)
                    // .first.split("\t")

    // 1.2. Read weight steps.

    val weightStepsLines: Array[String] = SparkUtils.readSmallFile(sc, weightStepsFile)
    val weights: Array[Double] = weightStepsLines.zipWithIndex.filter(_._2 % 2 == 0).map(_._1.toDouble)
    val thresholds: Array[Double] = weightStepsLines.zipWithIndex.filter(_._2 % 2 == 1).map(_._1.toDouble)

    // 1.3. Read data.

    val samples: RDD[String] = sc.textFile(dataFile)

    // 1.4. Generate weighted samples.

    val weightedSamples: RDD[String] = samples.map(sample => {
        val response: Double = sample.split("\t")(0).toDouble
        var sampleWeight: Double = weights(weights.length - 1)
        for (s <- 0 to thresholds.length - 1) {
          if (response < thresholds(s)) {
            sampleWeight = weights(s)
          }
        }
        sample + "\t" + sampleWeight
      })

    // 1.5. Save weighted data and header.

    weightedSamples.saveAsTextFile(weightedDataFile)
    sc.parallelize(features.toArray ++ Array("sample_weight"), 1).
      saveAsTextFile(weightedDataHeaderFile)

    sc.stop

  }

}
