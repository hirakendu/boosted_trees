package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel

import boosted_trees.Node


object SparkRegressionTreeModelTrainer {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var sparkMaster : String = SparkDefaultParameters.sparkMaster
		var sparkHome : String = SparkDefaultParameters.sparkHome
		val sparkAppName : String = SparkDefaultParameters.sparkAppName
		var sparkAppJars : String = SparkDefaultParameters.sparkAppJars
		var headerFile : String = SparkDefaultParameters.headerFile
		var featureWeightsFile : String = ""
		var dataFile : String = SparkDefaultParameters.trainDataFile
		var indexesDir : String = SparkDefaultParameters.indexesDir
		var indexedDataFile : String = SparkDefaultParameters.indexedTrainDataFile
		var modelDir : String = SparkDefaultParameters.treeModelDir
		var maxDepth : Int = 5
		var minGainFraction : Double = 0.01
		var minLocalGainFraction : Double = 0.1
		var minDistributedSamples : Int = 10000
		var useSampleWeights : Int = 0
		var useIndexedData : Int = 0
		var saveIndexedData : Int = 0
		var useCache : Int = 1
		var useArrays : Int = 1

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
			} else if (("--min-distributed-samples".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				minDistributedSamples = xargs(argi).toInt
			} else if (("--use-sample-weights".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				useSampleWeights = xargs(argi).toInt
			} else if (("--use-indexed-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				useIndexedData = xargs(argi).toInt
			} else if (("--save-indexed-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				saveIndexedData = xargs(argi).toInt
			} else if (("--use-cache".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				useCache = xargs(argi).toInt
			} else if (("--use-arrays".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				useArrays = xargs(argi).toInt
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
		val sc : SparkContext = new SparkContext(sparkMaster, sparkAppName,
				sparkHome, sparkAppJarsSeq)
		
		
		// 1. Read input data and index it.
		
		println("\n  Reading and indexing data.\n")
		
		// 1.1. Read header.
		val features : Array[String] = SparkUtils.readSmallFile(sc, headerFile)
										// .first.split("\t")
		val numFeatures : Int = features.length
		val featureTypes : Array[Int] = features.map(field => {if (field.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		var featureWeights : Array[Double] = Range(0, features.length).map(x => 1.0).toArray
		if (!featureWeightsFile.equals("")) {
			featureWeights = SparkUtils.readSmallFile(sc, featureWeightsFile).map(_.toDouble)
		}
		
		// 1.2 Read data and index it.
		
		var indexes : Array[Map[String, Int]] = null
		var samples : RDD[Array[Double]] = null
		if (useIndexedData == 0) {
			val rawSamples : RDD[String] = sc.textFile(dataFile)
			// Index categorical features/fields and re-encode data.
			indexes = SparkIndexing.generateIndexes(rawSamples, featureTypes)
			samples = SparkIndexing.indexRawData(rawSamples, featureTypes, indexes)
			
			// Save indexes and indexed data.
			SparkIndexing.saveIndexes(sc, indexesDir, features, indexes)
			if (saveIndexedData == 1) {
				SparkIndexing.saveIndexedData(indexedDataFile, samples, featureTypes)
			}
		} else {
			// Use indexed data.
			samples = SparkIndexing.readIndexedData(sc, indexedDataFile)
			// Read indexes for categorical features.
			indexes = SparkIndexing.readIndexes(sc, indexesDir, features)
		}
		val numValuesForFeatures : Array[Int] = new Array(numFeatures)
		for (j <- 0 to numFeatures - 1) {
			if (featureTypes(j) == 1) {
				numValuesForFeatures(j) = indexes(j).size
			}
		}
		
		
		// 2. Train tree model.
		
		println("\n  Training tree model.\n")
		
//		if (useCache == 1) {
//			// samples.persist(StorageLevel.MEMORY_AND_DISK)
//			samples.persist(StorageLevel.MEMORY_AND_DISK_SER)
//			// samples.persist
//			// samples.foreach(sample => {})  // Load now.
//		}
		
		val rootNode : Node = SparkRegressionTree.trainTree(samples, featureTypes,
				numValuesForFeatures, featureWeights, maxDepth, minGainFraction,
				minLocalGainFraction, minDistributedSamples, useSampleWeights,
				useArrays, useCache)
		
		
		// 3. Print and save the tree.
		
		println("\n  Saving the tree.\n")
		
		SparkRegressionTree.saveTree(sc, modelDir + "/nodes.txt", rootNode)
		SparkRegressionTree.printTree(sc, modelDir + "/tree.txt", rootNode)
		
	}

}
