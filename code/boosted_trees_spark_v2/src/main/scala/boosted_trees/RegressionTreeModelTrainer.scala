package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source

object RegressionTreeModelTrainer {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var headerFile : String = DefaultParameters.headerFile
		var featureWeightsFile : String = ""
		var dataFile : String =  DefaultParameters.trainDataFile
		var indexesDir : String =  DefaultParameters.indexesDir
		var indexedDataFile : String =  DefaultParameters.indexedTrainDataFile
		var modelDir : String = DefaultParameters.treeModelDir
		var maxDepth : Int = 5
		var minGainFraction : Double = 0.01
		var minLocalGainFraction : Double = 1
		var useSampleWeights : Int = 0
		var useIndexedData : Int = 0
		var saveIndexedData : Int = 0

		// 0.1. Read parameters.
		
		var xargs : Array[String] = args
		if (args.length == 1) {
			xargs = args(0).split("\\^")
		}
		var argi : Int = 0
		while (argi < xargs.length) {
			if (("--header-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				headerFile = xargs(argi)
			} else if (("--feature-weights-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				featureWeightsFile = xargs(argi)
			} else if (("--data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				dataFile = xargs(argi)
			} else if (("--indexes-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				indexesDir = xargs(argi)
			} else if (("--indexed-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				indexedDataFile = xargs(argi)
			} else if (("--model-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				modelDir = xargs(argi)
			} else if (("--max-depth".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				maxDepth = xargs(argi).toInt
			} else if (("--min-gain-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				minGainFraction = xargs(argi).toDouble
			} else if (("--min-local-gain-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				minLocalGainFraction = xargs(argi).toDouble
			} else if (("--use-sample-weights".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				useSampleWeights = xargs(argi).toInt
			} else if (("--use-indexed-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				useIndexedData = xargs(argi).toInt
			} else if (("--save-indexed-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				saveIndexedData = xargs(argi).toInt
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
		val features : Array[String] = Source.fromFile(new File(headerFile)).getLines.toArray
											// .first.split("\t")
		val numFeatures : Int = features.length
		val featureTypes : Array[Int] = features.map(field => {if (field.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		var featureWeights : Array[Double] = Range(0, features.length).map(x => 1.0).toArray
		if (!featureWeightsFile.equals("")) {
			featureWeights = Source.fromFile(new File(featureWeightsFile)).getLines.toArray.map(_.toDouble)
		}
		
		// 1.2 Read data and index it.
		
		var indexes : Array[Map[String, Int]] = null
		var samples : Array[Array[Double]] = null
		if (useIndexedData == 0) {
			// Index categorical features/fields and re-encode data.
			indexes = Indexing.generateIndexes(dataFile, featureTypes)
			samples = Indexing.indexRawData(dataFile, featureTypes, indexes)

			// Save indexes and indexed data.
			Indexing.saveIndexes(indexesDir, features, indexes)
			if (saveIndexedData == 1) {
				Indexing.saveIndexedData(indexedDataFile, samples, featureTypes)
			}
		} else {
			// Use indexed data.
			samples = Indexing.readIndexedData(indexedDataFile)
			// Read indexes for categorical features.
			indexes = Indexing.readIndexes(indexesDir, features)
		}
		val numValuesForFeatures : Array[Int] = new Array(numFeatures)
		for (j <- 1 to numFeatures - 1) {
			if (featureTypes(j) == 1) {
				numValuesForFeatures(j) = indexes(j).size
			}
		}
		
		// 2. Train tree model.
		
		println("\n  Training tree model.\n")
		
		val rootNode : Node = RegressionTree.trainTree(samples, featureTypes,
				numValuesForFeatures, featureWeights,
				maxDepth, minGainFraction, minLocalGainFraction,
				useSampleWeights)
		
		
		// 3. Print and save the tree.
		
		println("\n  Saving the tree.\n")
		
		RegressionTree.saveTree(modelDir + "/nodes.txt", rootNode)
		RegressionTree.printTree(modelDir + "/tree.txt", rootNode)
		
	}

}
