package boosted_trees.local

import java.io.File
import java.io.PrintWriter

import scala.io.Source

import boosted_trees.LabeledPoint

import boosted_trees.GBDTAlgorithmParameters
import boosted_trees.Node
import boosted_trees.loss._
import boosted_trees.GBDTModel


object GBDTModelTrainer {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    var headerFile: String = DefaultParameters.headerFile
    var dataFile: String = DefaultParameters.trainDataFile
    var modelDir: String = DefaultParameters.forestModelDir
    var lossFunction: String = "square"
    var numTrees: Int = 5
    var shrinkage: Double = 0.8
    var maxDepth: Int = 4
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
      } else if (("--num-trees".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        numTrees = xargs(argi).toInt
      } else if (("--shrinkage".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        shrinkage = xargs(argi).toDouble
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


    // 2. Train forest model.

    println("\n  Training forest model.\n")

    val gbdtParams: GBDTAlgorithmParameters =
      GBDTAlgorithmParameters(
        featureTypes,
        numTrees,
        shrinkage,
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
        maxNumQuantileSamples = 1,  // Not applicable.
        histogramsMethod,
        fastTree,
        batchSize,
        persistInterval = 1,  // Not applicable.
        numReducersPerNode = 0,  // Not applicable.
        maxNumReducers = 0  // Not applicable.
      )

    var model: GBDTModel = null
    if (lossFunction.equals("entropy")) {
      val algorithm = new GBDTAlgorithm[EntropyLossStats](
        new EntropyLoss, gbdtParams)
      model = algorithm.train(samples)
    } else {  // "square"
      val algorithm = new GBDTAlgorithm[SquareLossStats](
        new SquareLoss, gbdtParams)
      model = algorithm.train(samples)
    }


    // 3. Print and save the forest.

    println("\n  Saving the forest.\n")

    model.loadNamesForIds(features, indexes)
    (new File(modelDir)).mkdirs
    var printWriter: PrintWriter = new PrintWriter(new File(modelDir + "/forest.txt"))
    printWriter.println(model.explain().mkString("\n"))
    printWriter.close
    printWriter = new PrintWriter(new File(modelDir + "/forest.json"))
    printWriter.println(model.save().mkString("\n"))
    printWriter.close
    printWriter = new PrintWriter(new File(modelDir + "/forest_ext.json"))
    printWriter.println(model.saveExtended().mkString("\n"))
    printWriter.close
    var linesForTrees = model.saveSimple()
    (new File(modelDir + "/forest_nodes")).mkdirs
    printWriter = new PrintWriter(new File(modelDir + "/forest_nodes/num_trees.txt"))
    printWriter.println(numTrees)
    printWriter.close
    for (m <- 0 to numTrees - 1) {
      printWriter = new PrintWriter(new File(modelDir + "/forest_nodes/nodes_" + m + ".txt"))
      printWriter.println(linesForTrees(m).mkString("\n"))
      printWriter.close
    }

    linesForTrees = model.printDot()
    (new File(modelDir + "/forest_dot")).mkdirs
    printWriter = new PrintWriter(new File(modelDir + "/forest_dot/num_trees.txt"))
    printWriter.println(numTrees)
    printWriter.close
    for (m <- 0 to numTrees - 1) {
      printWriter = new PrintWriter(new File(modelDir + "/forest_dot/tree_" + m + ".dot"))
      printWriter.println(linesForTrees(m).mkString("\n"))
      printWriter.close
    }

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
