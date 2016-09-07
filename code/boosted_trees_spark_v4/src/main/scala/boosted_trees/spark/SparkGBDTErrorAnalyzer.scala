package boosted_trees.spark

import scala.collection.mutable.MutableList

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.scheduler.{SplitInfo, InputFormatInfo}
import org.apache.spark.deploy.SparkHadoopUtil

import boosted_trees.GBDTModel
import boosted_trees.Utils


object SparkGBDTErrorAnalyzer {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var headerFile: String = SparkDefaultParameters.headerFile
    var dataFile: String = SparkDefaultParameters.testDataFile
    var modelDir: String = SparkDefaultParameters.forestModelDir
    var errorDir: String = modelDir + "/error"
    var binaryMode: Int = 0
    var threshold: Double = 0.5
    var fullAuc: Int = 0
    var maxNumSummarySamples: Int = 100000
    var numReducers: Int = 10
    var useCache: Int = 0
    var rngSeed: Long = 42
    var waitTimeMs: Long = 1

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
      } else if (("--error-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        errorDir = xargs(argi)
      } else if (("--binary-mode".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        binaryMode = xargs(argi).toInt
      } else if (("--threshold".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        threshold = xargs(argi).toDouble
      } else if (("--full-auc".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        fullAuc = xargs(argi).toInt
      } else if (("--max-num-summary-samples".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxNumSummarySamples = xargs(argi).toInt
      } else if (("--num-reducers".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        numReducers = xargs(argi).toInt
      } else if (("--use-cache".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        useCache = xargs(argi).toInt
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


    // 1. Read model and test data.

    println("\n  Reading model and data.\n")

    // 1.1. Read forest model.

    val model: GBDTModel = new GBDTModel
    model.load(sc.textFile(modelDir + "/forest.json").collect)
    // model.loadExt(sc.textFile(modelDir + "/forest_ext.json").collect)
    // val numTrees: Int = sc.textFile(modelDir + "/forest_nodes/num_trees.txt").first.toInt
    // val linesForTrees: Array[Array[String]] = new Array(numTrees)
    // for (m <- 0 to numTrees - 1) {
    //   linesForTrees(m) = sc.textFile(modelDir + "/forest_nodes/nodes_" + m + ".txt").collect
    // }
    // model.loadSimple(linesForTrees)

    // 1.2. Read header.

    val features: Array[String] = SparkUtils.readSmallFile(sc, headerFile).drop(1)
                    // .first.split("\t").drop(1)
    val numFeatures: Int = features.length
    val featureTypes: Array[Int] = features.map(feature => {
        if (feature.endsWith("$")) 1 else if (feature.endsWith("#")) -1 else 0})
        // 0 -> continuous, 1 -> discrete, -1 ignored

    // 1.3. Read data.

    val testSamples: RDD[Array[String]] = sc.textFile(dataFile).map(_.split("\t", -1)).
        filter(_.length == featureTypes.length + 1)


    // 2. Predict using forest model and evaluate error.

    println("\n  Predicting on test data and analyzing error.\n")

    var predictedActual: RDD[(Double, Double)] = testSamples.map(testSample =>
        (model.predict(testSample.drop(1)), testSample(0).toDouble))
    if (binaryMode == 0) {
      predictedActual.map(x => "%.5f".format(x._1) + "\t" + "%.5f".format(x._2)).
          saveAsTextFile(errorDir + "/scatter_plot.txt")
    } else if (binaryMode == 1) {
      predictedActual.map(x => "%.5f".format(x._1) + "\t" + x._2.toInt).
          saveAsTextFile(errorDir + "/scatter_plot.txt")
    }
    predictedActual = sc.textFile(errorDir + "/scatter_plot.txt").
        map(_.split("\t", -1)).map(x => (x(0).toDouble, x(1).toDouble))
    if (useCache == 1) {
      predictedActual.persist(StorageLevel.MEMORY_AND_DISK)
      // predictedActual.persist(StorageLevel.MEMORY_AND_DISK_2)
      // predictedActual.persist(StorageLevel.MEMORY_AND_DISK_SER)
      // predictedActual.persist
      // predictedActual.foreach(sample => {})  // Load now.
    }

    val errorStats: (Long, Double, Double) = predictedActual.
        map(x => (1L, (x._1 - x._2) * (x._1 - x._2),math.abs(x._1 - x._2))).
        reduce((stats1, stats2) => (stats1._1 + stats2._1,
            stats1._2 + stats2._2, stats1._3 + stats2._3))
    val trivialResponse: Double = model.trees.map(_.rootNode.response).sum
    val trivialErrorStats: (Long, Double, Double) = predictedActual.map(x => {
          (1L, (trivialResponse - x._2) * (trivialResponse - x._2),
            math.abs(trivialResponse - x._2))
        }).reduce((stats1, stats2) => (stats1._1 + stats2._1,
            stats1._2 + stats2._2, stats1._3 + stats2._3))

    var binaryErrorStats: (Long, Long, Long, Long, Double) = null
    if (binaryMode == 1) {
      binaryErrorStats = predictedActual.map(x => {
          val predicted: Int = if (x._1 < threshold) 0 else 1
          val stats: (Long, Long, Long, Long, Double) = x._2.toInt match {
            case 0 => (1L, predicted, 0L, 0, if (x._1 < 1-1e-10) math.log(1 - x._1) else -10.0)
            case 1 => (0L, 0, 1L, 1 - predicted, if (x._1 > 1e-10) math.log(x._1) else -10.0)
          }
          stats
        }).reduce((stats1, stats2) => (stats1._1 + stats2._1,
            stats1._2 + stats2._2, stats1._3 + stats2._3,
            stats1._4 + stats2._4, stats1._5 + stats2._5))
    }

    // 2.1. AUC for binary case.

    var roc: RDD[(Double, Double, Double)] = null
    var auc: Double = -1
    if (binaryMode == 1 && fullAuc == 1) {
      val rocAuc = SparkUtils.findRocAuc(predictedActual.map(x => (x._1, x._2.toInt)), numReducers)
      roc = rocAuc._1
      auc = rocAuc._2
    }

    // 2.2. Summary samples.

    val predictedActualSamples: Array[(Double, Double)] =
        predictedActual.takeSample(false, maxNumSummarySamples, rngSeed)

    var rocSample: Array[(Double, Double, Double)] = null
    var aucSample: Double = 0
    if (binaryMode == 1) {
      val rocAuc = Utils.findRocAuc(predictedActualSamples.map(x => (x._1, x._2.toInt)))
      rocSample = rocAuc._1
      aucSample = rocAuc._2
    }


    // 3. Save error statistics.

    println("\n  Saving error statistics.\n")

    val lines: MutableList[String] = MutableList()
    if (binaryMode == 1) {
      val fp: Long = binaryErrorStats._2
      val fn: Long = binaryErrorStats._4
      val tp: Long = binaryErrorStats._3 - fn
      val tn: Long = binaryErrorStats._1 - fp
      val ll: Double = binaryErrorStats._5 / ((tn + tp + fn + fp) * math.log(2))
      val pAvg: Double = (tp + fn).toDouble / (tn + tp + fn + fp)
      val entropy: Double =  - ((pAvg * math.log(pAvg)) + ((1-pAvg) * math.log(1-pAvg))) / math.log(2)
      lines += "TPR = Recall = " + tp + "/" + (tp + fn) + " = " +
          "%.5f".format(tp.toDouble / (tp + fn))
      lines += "FPR = " + fp + "/" + (tn + fp) + " = " +
          "%.5f".format(fp.toDouble / (tn + fp))
      lines += "Precision = " + tp + "/" + (tp + fp) + " = " +
          "%.5f".format(tp.toDouble / (tp + fp))
      lines += "F1 = " +  "%.5f".format(2 * tp.toDouble / (2 * tp + fn + fp))
      lines += "A = " + (tn + tp) + "/" + (tn + tp + fn + fp) + " = " +
          "%.5f".format((tn + tp).toDouble / (tn + tp + fn + fp))
      lines += "NLL = " + "%.5f".format(-ll)
      lines += "Entropy = H(" + "%.5f".format(pAvg) + ") = " + "%.5f".format(entropy)
      lines += "RIG = " + "%.5f".format(1.0 + (ll/entropy))
      lines += "AUC = " + "%.5f".format(auc)
      lines += "Sample AUC = " + "%.5f".format(aucSample)
    }
    lines += "RMSE = " + "%.5f".format(math.sqrt(errorStats._2 / errorStats._1))
    lines += "MAE = " + "%.5f".format(errorStats._3 / errorStats._1)
    lines += "Trivial response = " + "%.5f".format(trivialResponse)
    lines += "Trivial RMSE = " + "%.5f".format(math.sqrt(trivialErrorStats._2 / trivialErrorStats._1))
    lines += "Trivial MAE = " + "%.5f".format(trivialErrorStats._3 / trivialErrorStats._1)
    sc.parallelize(lines, 1).saveAsTextFile(errorDir + "/error.txt")
    if (binaryMode == 0) {
      sc.parallelize(predictedActualSamples.map(x => "%.5f".format(x._1) + "\t" +
          "%.5f".format(x._2)), 1).saveAsTextFile(errorDir + "/scatter_plot.txt.sample")
    } else if (binaryMode == 1) {
      sc.parallelize(predictedActualSamples.map(x => "%.5f".format(x._1) + "\t" +
          x._2.toInt), 1).saveAsTextFile(errorDir + "/scatter_plot.txt.sample")
      sc.parallelize(rocSample.map(x => "%.5f".format(x._1) + "\t" +
        "%.5f".format(x._2) + "\t" + "%.5f".format(x._3)), 1).
        saveAsTextFile(errorDir + "/roc.txt.sample")
      if (fullAuc == 1) {
        roc.map(x => "%.5f".format(x._1) + "\t" +
          "%.5f".format(x._2) + "\t" + "%.5f".format(x._3)).
          saveAsTextFile(errorDir + "/roc.txt")
      }
    }

    sc.stop

  }

}
