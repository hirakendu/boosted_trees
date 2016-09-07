package boosted_trees.spark

import scala.collection.mutable.MutableList

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.scheduler.{SplitInfo, InputFormatInfo}
import org.apache.spark.deploy.SparkHadoopUtil


object SparkClassificationErrorAnalyzer {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    val sparkAppName: String = SparkDefaultParameters.sparkAppName
    var headerFile: String =  ""  // DefaultParameters.workDir + "/header.txt"
    var dataFile: String = SparkDefaultParameters.workDir + "/scores_labels.txt"
    var outputDir: String = SparkDefaultParameters.workDir + "/error"
    var scoreField: String = "score"
    var labelField: String = "label"
    var threshold: Double = 0.5
    var numReducers: Int = 10
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
      } else if (("--output-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        outputDir = xargs(argi)
      } else if (("--score-field".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        scoreField = xargs(argi)
      } else if (("--label-field".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        labelField = xargs(argi)
      } else if (("--threshold".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        threshold = xargs(argi).toDouble
      } else if (("--num-reducers".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        numReducers = xargs(argi).toInt
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


    // 1. Schema.

    var scoreFieldId: Int = 0
    var labelFieldId: Int = 1
    var allFieldsCount: Int = 2
    if (!"".equals(headerFile)) {
      val allFields: Array[String] = sc.textFile(headerFile).collect.map(_.toLowerCase)
      val allFieldIds: Map[String, Int] = allFields.zipWithIndex.toMap
      allFieldsCount = allFields.length
      scoreFieldId = allFieldIds(scoreField.toLowerCase)
      labelFieldId = allFieldIds(labelField.toLowerCase)
    }


    // 2. Find ROC and AUC from scatter-plot data.

    val scoresLabels: RDD[(Double, Int)] = sc.textFile(dataFile).
        map(_.split("\t")).
        filter(_.length == allFieldsCount).
        filter(record => !"".equals(record(scoreFieldId)) &&
            !"".equals(record(labelFieldId))).
        map(record => (record(scoreFieldId).toDouble, record(labelFieldId).toDouble.toInt))

    val errorStats: (Long, Long, Long, Long, Double) = scoresLabels.
    map(x => {
      val predicted: Int = if (x._1 < threshold) 0 else 1
      val stats: (Long, Long, Long, Long, Double) = x._2.toInt match {
        case 0 => (1L, predicted, 0L, 0, if (x._1 < 1-1e-10) math.log(1 - x._1) else -10.0)
        case 1 => (0L, 0, 1L, 1 - predicted, if (x._1 > 1e-10) math.log(x._1) else -10.0)
      }
      stats
    }).
    reduce((stats1, stats2) => (stats1._1 + stats2._1,
        stats1._2 + stats2._2, stats1._3 + stats2._3,
        stats1._4 + stats2._4, stats1._5 + stats2._5))

    val rocAuc: (RDD[(Double, Double, Double)], Double) = SparkUtils.findRocAuc(scoresLabels, numReducers)

    rocAuc._1.map(x => "%.5f".format(x._1) + "\t" + "%.5f".format(x._2) + "\t" +
        "%.5f".format(x._3)).
        // coalesce(outputParts, shuffle = true).
        saveAsTextFile(outputDir + "/roc.txt")

    val lines: MutableList[String] = MutableList()
    val fp: Long = errorStats._2
    val fn: Long = errorStats._4
    val tp: Long = errorStats._3 - fn
    val tn: Long = errorStats._1 - fp
    val ll: Double = errorStats._5 / ((tn + tp + fn + fp) * math.log(2))
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
    lines += "AUC = " + "%.5f".format(rocAuc._2)

    sc.parallelize(lines, 1).saveAsTextFile(outputDir + "/error.txt")

    sc.stop

  }

}
