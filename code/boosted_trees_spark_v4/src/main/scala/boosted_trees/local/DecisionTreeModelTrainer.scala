package boosted_trees.local

import java.io.File
import java.io.PrintWriter

import scala.io.Source

import boosted_trees.DecisionTreeAlgorithmParameters
import boosted_trees.Node
import boosted_trees.LabeledPoint
import boosted_trees.loss._
import boosted_trees.DecisionTreeModel


object DecisionTreeModelTrainer {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    var headerFile: String = DefaultParameters.headerFile
    var dataFile: String = DefaultParameters.trainDataFile
    var modelDir: String = DefaultParameters.treeModelDir
    var lossFunction: String = "square"
    var maxDepth: Int = 5
    var minGainFraction: Double = 0.01
    var minLocalGainFraction: Double = 1
    var minCount: Int = 2
    var minWeight: Double = 2
    var featureWeightsFile: String = ""
    var useSampleWeights: Int = 0
    var histogramsMethod: String = "array-aggregate"
    var useGlobalQuantiles: Int = 1
    var fastTree: Int = 1
    var batchSize: Int = 16
    var useEncodedData: Int = 0
    var encodedDataFile: String = DefaultParameters.encodedTrainDataFile
    var dictsDir: String = DefaultParameters.dictsDir
    var saveEncodedData: Int = 0

    var maxNumQuantiles: Int = 1000
    var maxNumQuantileSamples: Int = 100000

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
      } else if (("--loss-function".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        lossFunction = xargs(argi)
      } else if (("--max-depth".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxDepth = xargs(argi).toInt
      } else if (("--min-gain-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minGainFraction = xargs(argi).toDouble
      } else if (("--min-local-gain-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minLocalGainFraction = xargs(argi).toDouble
      } else if (("--min-count".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minCount = xargs(argi).toInt
      } else if (("--min-weight".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        minWeight = xargs(argi).toDouble
      } else if (("--feature-weights-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        featureWeightsFile = xargs(argi)
      } else if (("--use-sample-weights".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        useSampleWeights = xargs(argi).toInt
      } else if (("--histograms-method".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        histogramsMethod = xargs(argi)
      } else if (("--use-global-quantiles".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        useGlobalQuantiles = xargs(argi).toInt
      } else if (("--fast-tree".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        fastTree = xargs(argi).toInt
      } else if (("--batch-size".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        batchSize = xargs(argi).toInt
      } else if (("--use-encoded-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        useEncodedData = xargs(argi).toInt
      } else if (("--encoded-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        encodedDataFile = xargs(argi)
      } else if (("--dicts-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        dictsDir = xargs(argi)
      } else if (("--save-encoded-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        saveEncodedData = xargs(argi).toInt
      } else {
        println("\n  Error parsing argument \"" + xargs(argi) +
            "\".\n")
        return
      }
      argi += 1
    }


    // 1. Read input data and index it.

    println("\n  Reading and indexing data.\n")

    // 1.1. Read header.

    val features: Array[String] = Source.fromFile(new File(headerFile)).getLines.toArray.drop(1)
                    // .first.split("\t").drop(1)
    val numFeatures: Int = features.length
    val featureTypes: Array[Int] = features.map(feature => {
        if (feature.endsWith("$")) 1 else if (feature.endsWith("#")) -1 else 0})
        // 0 -> continuous, 1 -> discrete, -1 ignored
    var featureWeights: Array[Double] = Array.fill[Double](numFeatures)(1.0)
    if (!featureWeightsFile.equals("")) {
      featureWeights = Source.fromFile(new File(featureWeightsFile)).getLines.toArray.
          drop(1).map(_.toDouble)
    }

    // 1.2 Read data and encode it.

    var indexes: Array[Map[String, Int]] = null
    var quantilesForFeatures: Array[Array[Double]] = null
    var samples: Array[LabeledPoint] = null
    if (useEncodedData == 0) {
      val rawSamples: Array[String] = Source.fromFile(new File(dataFile)).getLines.toArray
      // Index categorical features, find quantiles for continuous features
      // and encode data.
      indexes = DataEncoding.generateIndexes(rawSamples, featureTypes)
      samples = DataEncoding.encodeRawData(rawSamples, featureTypes, indexes)
      quantilesForFeatures = DataEncoding.findQuantilesForFeatures(
          samples, featureTypes, maxNumQuantiles)

      // Save dicts and encoded data.
      if (saveEncodedData == 1) {
        DataEncoding.saveIndexes(dictsDir, features, indexes)
        DataEncoding.saveQuantiles(dictsDir, features, quantilesForFeatures)
        DataEncoding.saveEncodedData(encodedDataFile, samples, featureTypes)
      }
    } else {
      // Use encoded data.
      samples = DataEncoding.readEncodedData(encodedDataFile)
      // Read indexes for categorical features.
      indexes = DataEncoding.readIndexes(dictsDir, features)
      // Read quantiles for continuous features.
      quantilesForFeatures = DataEncoding.readQuantiles(dictsDir, features)
    }
    val cardinalitiesForFeatures: Array[Int] = new Array(numFeatures)
    for (j <- 0 to numFeatures - 1) {
      if (featureTypes(j) == 1) {
        cardinalitiesForFeatures(j) = indexes(j).size
      }
    }


    // 2. Train tree model.

    println("\n  Training tree model.\n")

    val dtParams: DecisionTreeAlgorithmParameters =
      DecisionTreeAlgorithmParameters(
        featureTypes,
        maxDepth,
        minGainFraction,
        minLocalGainFraction,
        minWeight,
        minCount,
        featureWeights,
        useSampleWeights,
        useCache = 0,  // Not applicable.
        cardinalitiesForFeatures,
        useGlobalQuantiles,
        quantilesForFeatures,
        maxNumQuantiles,
        maxNumQuantileSamples,
        histogramsMethod,
        batchSize,
        numReducersPerNode = 0,  // Not applicable.
        maxNumReducers = 0  // Not applicable.
      )

    var model: DecisionTreeModel = null
    if (fastTree == 1) {
      if (lossFunction.equals("entropy")) {
        val algorithm = new FastDecisionTreeAlgorithm[EntropyLossStats](
          new EntropyLoss, dtParams)
        model = algorithm.train(samples)
      } else {  // "square"
        val algorithm = new FastDecisionTreeAlgorithm[SquareLossStats](
          new SquareLoss, dtParams)
        model = algorithm.train(samples)
      }
    } else {  // fastTree == 0
      if (lossFunction.equals("entropy")) {
        val algorithm = new SimpleDecisionTreeAlgorithm[EntropyLossStats](
          new EntropyLoss, dtParams)
        model = algorithm.train(samples)
      } else {  // "square"
        val algorithm = new SimpleDecisionTreeAlgorithm[SquareLossStats](
          new SquareLoss, dtParams)
        model = algorithm.train(samples)
      }
    }


    // 3. Print and save the tree.

    println("\n  Saving the tree.\n")

    model.loadNamesForIds(features, indexes)
    (new File(modelDir)).mkdirs
    var printWriter: PrintWriter = new PrintWriter(new File(modelDir + "/tree.txt"))
    printWriter.println(model.explain().mkString("\n"))
    printWriter.close
    printWriter = new PrintWriter(new File(modelDir + "/tree.json"))
    printWriter.println(model.save().mkString("\n"))
    printWriter.close
    printWriter = new PrintWriter(new File(modelDir + "/tree_ext.json"))
    printWriter.println(model.saveExtended().mkString("\n"))
    printWriter.close
    printWriter = new PrintWriter(new File(modelDir + "/nodes.txt"))
    printWriter.println(model.saveSimple().mkString("\n"))
    printWriter.close

    printWriter = new PrintWriter(new File(modelDir + "/tree.dot"))
    printWriter.println(model.printDot().mkString("\n"))
    printWriter.close

    printWriter = new PrintWriter(new File(modelDir + "/feature_importances.txt"))
    printWriter.println(model.evaluateFeatureImportances().
        map(x => x._1 + "\t" + "%.2f".format(x._2)).mkString("\n"))
    printWriter.close
    printWriter = new PrintWriter(new File(modelDir + "/feature_subset_importances.txt"))
    printWriter.println(model.evaluateFeatureSubsetImportances().
        map(x => x._1.mkString(",") + "\t" + "%.2f".format(x._2)).mkString("\n"))
    printWriter.close

  }

}
