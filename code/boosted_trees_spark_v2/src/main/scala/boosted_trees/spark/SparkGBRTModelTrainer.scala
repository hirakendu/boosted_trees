package boosted_trees.spark

import spark.RDD
import spark.SparkContext
import spark.SparkContext._
import spark.storage.StorageLevel

import boosted_trees.Node


object SparkGBRTModelTrainer {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var sparkMaster : String = SparkDefaultParameters.sparkMaster
		var sparkHome : String = SparkDefaultParameters.sparkHome
		var sparkAppJars : String = SparkDefaultParameters.sparkAppJars
		var headerFile : String = SparkDefaultParameters.headerFile
		var dataFile : String = SparkDefaultParameters.trainDataFile
		var indexesDir : String = SparkDefaultParameters.indexesDir
		var indexedDataFile : String = SparkDefaultParameters.indexedTrainDataFile
		var modelDir : String = SparkDefaultParameters.forestModelDir
		var numTrees : Int = 5
		var shrinkage : Double = 0.8
		var maxDepth : Int = 4
		var minGainFraction : Double = 0.01
		var minDistributedSamples : Int = 10000
		var initialNumTrees : Int = 0
		var useIndexedData : Int = 0
		var saveIndexedData : Int = 0
		var cacheIndexedData : Int = 0

		// 0.1. Read parameters.
			
		if (System.getenv("SPARK_MASTER") != null) {
			sparkMaster = System.getenv("SPARK_MASTER")
		}
		if (System.getenv("SPARK_HOME") != null) {
			sparkHome = System.getenv("SPARK_HOME")
		}
		if (System.getenv("SPARK_APP_JARS") != null) {
			sparkAppJars = System.getenv("SPARK_APP_JARS")
		}
		var xargs : Array[String] = args
		if (args.length == 1) {
			xargs = args(0).split("\\^")
		}
		var argi : Int = 0
		while (argi < xargs.length) {
			if (("--spark-master".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				sparkMaster = xargs(argi)
			} else if (("--spark-home".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				sparkHome = xargs(argi)
			} else if (("--spark-app-jars".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				sparkAppJars = xargs(argi)
			} else if (("--header-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
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
			} else if (("--min-distributed-samples".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				minDistributedSamples = xargs(argi).toInt
			} else if (("--initial-num-trees".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				initialNumTrees = xargs(argi).toInt
			} else if (("--use-indexed-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				useIndexedData = xargs(argi).toInt
			} else if (("--save-indexed-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				saveIndexedData = xargs(argi).toInt
			} else if (("--cache-indexed-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				cacheIndexedData = xargs(argi).toInt
			} else {
				println("\n  Error parsing argument \"" + xargs(argi) +
						"\".\n")
				return
			}
			argi += 1
		}
		
		// 0.2. Create Spark context.
		var sparkAppJarsSeq : Seq[String] = Nil
		if (sparkAppJars != null) {
			sparkAppJarsSeq = sparkAppJars.split(",").toSeq
		}
		val sc : SparkContext = new SparkContext(sparkMaster, "Spark Regression Tree",
				sparkHome, sparkAppJarsSeq)
		
		
		// 1. Read input data and index it.
		
		println("\n  Reading and indexing data.\n")
		
		// 1.1. Read header.
		val features : Array[String] = SparkUtils.readSmallFile(sc, headerFile)
										// .first.split("\t")
		val featureTypes : Array[Int] = features.map(field => {if (field.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		
		// 1.2 Read data and index it.
		
		var samples : RDD[Array[Double]] = null
		
		if (useIndexedData == 0) {
			val rawSamples : RDD[String] = sc.textFile(dataFile)
			// Index categorical features/fields and re-encode data.
			val indexes :  Array[Map[String,Int]] = SparkIndexing.generateIndexes(rawSamples, featureTypes)
			samples = SparkIndexing.indexRawData(rawSamples, featureTypes, indexes)
			
			// Save indexes and indexed data.
			SparkIndexing.saveIndexes(sc, indexesDir, features, indexes)
			if (saveIndexedData == 1) {
				SparkIndexing.saveIndexedData(indexedDataFile, samples, featureTypes)
			}
		} else {
			// Use indexed data.
			samples = SparkIndexing.readIndexedData(sc, indexedDataFile)
		}
		
		// 2. Read initial forest model.
		
		var initialRootNodes : Array[Node] = Array() 
		if (initialNumTrees > 0) {
			initialRootNodes = SparkGBRT.readForest(sc, modelDir + "/nodes/")
		}
		
		// 3. Train forest model.
		
		println("\n  Training forest model.\n")
		
		if (cacheIndexedData == 1) {
			// samples.persist(StorageLevel.MEMORY_AND_DISK)
			// samples.persist
			samples.persist(StorageLevel.MEMORY_AND_DISK_SER)
		}
		
		val rootNodes : Array[Node] = SparkGBRT.trainForest(samples, featureTypes,
				numTrees, shrinkage, maxDepth, minGainFraction,
				minDistributedSamples, initialRootNodes)
		
		
		// 4. Print and save the tree.
		
		println("\n  Saving the forest.\n")
		
		SparkGBRT.saveForest(sc, modelDir + "/nodes/", rootNodes, initialNumTrees)
		SparkGBRT.printForest(sc, modelDir + "/trees/", rootNodes, initialNumTrees)
		
	}

}