package boosted_trees.local

import java.io.File
import java.io.PrintWriter

import scala.io.Source
import scala.collection.mutable.MutableList

import boosted_trees.DecisionTreeModel
import boosted_trees.Utils


object DecisionTreeScorer {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    var headerFile: String = DefaultParameters.headerFile
    var dataFile: String = DefaultParameters.testDataFile
    var modelDir: String = DefaultParameters.treeModelDir
    var outputFile: String = DefaultParameters.workDir + "/test_data_with_scores.txt"
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


    // 1. Read model and test data.

    // 1.1. Read tree model.

    val model: DecisionTreeModel = new DecisionTreeModel
    model.load(Source.fromFile(new File(modelDir + "/tree.json")).getLines.toArray)
    // model.loadExt(Source.fromFile(new File(modelDir + "/tree_ext.json")).getLines.toArray)
    // model.loadSimple(Source.fromFile(new File(modelDir + "/nodes.txt")).getLines.toArray)

    // 1.2. Read header.

    val features: Array[String] = Source.fromFile(new File(headerFile)).getLines.toArray.drop(1)
                    // .first.split("\t").drop(1)
    val numFeatures: Int = features.length
    val featureTypes: Array[Int] = features.map(feature => {
        if (feature.endsWith("$")) 1 else if (feature.endsWith("#")) -1 else 0})
        // 0 -> continuous, 1 -> discrete, -1 ignored

    // 1.3. Read data.

    var testSamples: Array[Array[String]] = Source.fromFile(new File(dataFile)).getLines.toArray.
        map(_.split("\t", -1)).filter(_.length == featureTypes.length + 1)


    // 2. Predict and save.

    val testSamplesWithScores: Array[(Array[String], Double)] =
        testSamples.map(testSample =>
          (testSample, model.predict(testSample.drop(1))))
    val printWriter: PrintWriter = new PrintWriter(new File(outputFile))
    if (binaryMode == 0) {
      printWriter.println(testSamplesWithScores.
          map(x => x._1.mkString("\t") + "\t" + "%.5f".format(x._2)).
          mkString("\n"))
    } else if (binaryMode == 1) {
      printWriter.println(testSamplesWithScores.
        map(x => {
          var b: Int = 0
          if (x._2 >= threshold) {
            b = 1
          }
          x._1.mkString("\t") + "\t" + b
        }).
        mkString("\n"))
    }
    printWriter.close

  }

}
