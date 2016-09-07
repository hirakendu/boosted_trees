package boosted_trees.spark

import scala.collection.mutable.MutableList

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.scheduler.{SplitInfo, InputFormatInfo}
import org.apache.spark.deploy.SparkHadoopUtil

import boosted_trees.DecisionTreeModel
import boosted_trees.Utils


object SparkDecisionTreeScorer {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var headerFile: String = SparkDefaultParameters.headerFile
    var dataFile: String = SparkDefaultParameters.testDataFile
    var modelDir: String = SparkDefaultParameters.treeModelDir
    var outputFile: String = SparkDefaultParameters.workDir + "/test_data_with_scores.txt"
    var binaryMode: Int = 0
    var threshold: Double = 0.5

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
      } else if (("--output-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        outputFile = xargs(argi)
      } else if (("--binary-mode".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        binaryMode = xargs(argi).toInt
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


    // 1. Read model, read and score test data and save.

    // 1.1. Read tree model.

    val model: DecisionTreeModel = new DecisionTreeModel
    model.load(sc.textFile(modelDir + "/tree.json").collect)
    // model.loadExt(sc.textFile(modelDir + "/tree_ext.json").collect)
    // model.loadSimple(sc.textFile(modelDir + "/nodes.txt").collect)

    // 1.2. Read header.

    val features: Array[String] = SparkUtils.readSmallFile(sc, headerFile).drop(1)
                    // .first.split("\t").drop(1)
    val numFeatures: Int = features.length
    val featureTypes: Array[Int] = features.map(feature => {
        if (feature.endsWith("$")) 1 else if (feature.endsWith("#")) -1 else 0})
        // 0 -> continuous, 1 -> discrete, -1 ignored

    // 1.3. Read data.

    var testSamples: RDD[Array[String]] = sc.textFile(dataFile).map(_.split("\t", -1)).
        filter(_.length == featureTypes.length + 1)


    // 2. Predict and save.

    var testSamplesWithScores: RDD[(Array[String], Double)] =
        testSamples.map(testSample =>
          (testSample, model.predict(testSample.drop(1))))
    if (binaryMode == 0) {
      testSamplesWithScores.map(x => x._1.mkString("\t") + "\t" + "%.5f".format(x._2)).
      saveAsTextFile(outputFile)
    } else if (binaryMode == 1) {
      testSamplesWithScores.
      map(x => {
        var b: Int = 0
        if (x._2 >= threshold) {
          b = 1
        }
        x._1.mkString("\t") + "\t" + b
      }).
      saveAsTextFile(outputFile)
    }

    sc.stop

  }

}
