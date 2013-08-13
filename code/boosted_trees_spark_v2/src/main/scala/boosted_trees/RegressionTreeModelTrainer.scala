package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source

object RegressionTreeModelTrainer {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var headerFile : String = DefaultParameters.headerFile
		var dataFile : String =  DefaultParameters.trainDataFile
		var indexesDir : String =  DefaultParameters.indexesDir
		var indexedDataFile : String =  DefaultParameters.indexedTrainDataFile
		var modelDir : String = DefaultParameters.treeModelDir
		var maxDepth : Int = 5
		var minGainFraction : Double = 0.01
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
		val featureTypes : Array[Int] = features.map(field => {if (field.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		
		// 1.2 Read data and index it.
		
		var samples : List[Array[Double]] = Nil
		
		if (useIndexedData == 0) {
			// Index categorical features/fields and re-encode data.
			val indexes :  Array[Map[String,Int]] =
				Indexing.generateIndexes(Source.fromFile(new File(dataFile)).getLines, featureTypes)
			samples = Indexing.indexRawData(Source.fromFile(new File(dataFile)).getLines, featureTypes, indexes)

			// Save indexes and indexed data.
			Indexing.saveIndexes(indexesDir, features, indexes)
			if (saveIndexedData == 1) {
				Indexing.saveIndexedData(indexedDataFile, samples, featureTypes)
			}
		} else {
			// Use indexed data.
			samples = Indexing.readIndexedData(indexedDataFile)
		}
		
		
		// 2. Train tree model.
		
		println("\n  Training tree model.\n")
		
		val rootNode : Node = RegressionTree.trainTree(samples, featureTypes, maxDepth, minGainFraction)
		
		
		// 3. Print and save the tree.
		
		println("\n  Saving the tree.\n")
		
		RegressionTree.saveTree(modelDir + "/nodes.txt", rootNode)
		RegressionTree.printTree(modelDir + "/tree.txt", rootNode)
		
	}


}