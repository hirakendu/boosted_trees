package boosted_trees.local

import java.io.File
import java.io.PrintWriter

import scala.io.Source
import scala.collection.mutable.MutableList

import boosted_trees.GBDTModel
import boosted_trees.Utils


object GBDTErrorAnalyzer {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    var headerFile: String = DefaultParameters.headerFile
    var dataFile: String = DefaultParameters.testDataFile
    var modelDir: String = DefaultParameters.forestModelDir
    var errorDir: String = modelDir + "/error"
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
      } else if (("--error-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        errorDir = xargs(argi)
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


    // 1. Read model and test data.

    println("\n  Reading model and data.\n")

    // 1.1. Read forest model.

    val model: GBDTModel = new GBDTModel
    model.load(Source.fromFile(new File(modelDir + "/forest.json")).getLines.toArray)
    // model.loadExt(Source.fromFile(new File(modelDir + "/forest_ext.json")).getLines.toArray)
    // val numTrees: Int = Source.fromFile(new File(modelDir + "/forest_nodes/num_trees.txt")).getLines.toArray.first.toInt
    // val linesForTrees: Array[Array[String]] = new Array(numTrees)
    // for (m <- 0 to numTrees - 1) {
    //   linesForTrees(m) = Source.fromFile(new File(modelDir + "/forest_nodes/nodes_" + m + ".txt")).getLines.toArray
    // }
    // model.loadSimple(linesForTrees)

    // 1.2. Read header.

    val features: Array[String] = Source.fromFile(new File(headerFile)).getLines.toArray.drop(1)
                    // .first.split("\t").drop(1)
    val numFeatures: Int = features.length
    val featureTypes: Array[Int] = features.map(feature => {
        if (feature.endsWith("$")) 1 else if (feature.endsWith("#")) -1 else 0})
        // 0 -> continuous, 1 -> discrete, -1 ignored

    // 1.3. Read data.

    val testSamples: Array[Array[String]] = Source.fromFile(new File(dataFile)).getLines.toArray.
        map(_.split("\t", -1)).filter(_.length == featureTypes.length + 1)


    // 2. Predict using forest model and evaluate error.

    println("\n  Predicting on test data and analyzing error.\n")

    val predictedActual: Array[(Double, Double)] = testSamples.map(testSample =>
          (model.predict(testSample.drop(1)), testSample(0).toDouble))

    val errorStats: (Long, Double, Double) = predictedActual.
        map(x => (1L, (x._1 - x._2) * (x._1 - x._2),math.abs(x._1 - x._2))).
        reduce((stats1, stats2) => (stats1._1 + stats2._1,
            stats1._2 + stats2._2, stats1._3 + stats2._3))
    val trivialResponse = model.trees.map(_.rootNode.response).sum
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


    // 3. Calculate AUC for binary case.

    var roc: Array[(Double, Double, Double)] = null
    var auc: Double = 0
    if (binaryMode == 1) {
      val scoresLabels: Array[(Double, Int)] = predictedActual.map(x => (x._1, x._2.toInt))
      val rocAuc = Utils.findRocAuc(scoresLabels)
      roc = rocAuc._1
      auc = rocAuc._2
    }


    // 4. Save error statistics.

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
    }
    lines += "RMSE = " + "%.5f".format(math.sqrt(errorStats._2 / errorStats._1))
    lines += "MAE = " + "%.5f".format(errorStats._3 / errorStats._1)
    lines += "Trivial response = " + "%.5f".format(trivialResponse)
    lines += "Trivial RMSE = " + "%.5f".format(math.sqrt(trivialErrorStats._2 / trivialErrorStats._1))
    lines += "Trivial MAE = " + "%.5f".format(trivialErrorStats._3 / trivialErrorStats._1)
    (new File(errorDir)).mkdirs
    var printWriter: PrintWriter = new PrintWriter(new File(errorDir + "/error.txt"))
    printWriter.println(lines.mkString("\n"))
    printWriter.close
    if (binaryMode == 0) {
      printWriter = new PrintWriter(new File(errorDir + "/scatter_plot.txt"))
      printWriter.println(predictedActual.map(x => "%.5f".format(x._1) + "\t" +
          "%.5f".format(x._2)).mkString("\n"))
      printWriter.close
    } else if (binaryMode == 1) {
      printWriter = new PrintWriter(new File(errorDir + "/scatter_plot.txt"))
      printWriter.println(predictedActual.map(x => "%.5f".format(x._1) + "\t" +
          x._2.toInt).mkString("\n"))
      printWriter.close
      printWriter = new PrintWriter(new File(errorDir + "/roc.txt"))
      printWriter.println(roc.map(x => "%.5f".format(x._1) + "\t" +
        "%.5f".format(x._2) + "\t" + "%.5f".format(x._3)).mkString("\n"))
      printWriter.close
    }

  }

}
