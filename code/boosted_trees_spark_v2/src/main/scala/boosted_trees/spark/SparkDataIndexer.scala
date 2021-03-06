package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object SparkDataIndexer {
	
	def main(args : Array[String]) : Unit = {

		// 0.0. Default parameters.
		var sparkMaster : String = SparkDefaultParameters.sparkMaster
		var sparkHome : String = SparkDefaultParameters.sparkHome
		val sparkAppName : String = SparkDefaultParameters.sparkAppName
		var sparkAppJars : String = SparkDefaultParameters.sparkAppJars
		var headerFile : String = SparkDefaultParameters.headerFile
		var dataFile : String = SparkDefaultParameters.trainDataFile
		var indexesDir : String = SparkDefaultParameters.indexesDir
		var indexedDataFile : String = SparkDefaultParameters.indexedTrainDataFile
		var generateIndexes : Int = 1
		var encodeData : Int = 1

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
			}  else if (("--generate-indexes".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				generateIndexes = xargs(argi).toInt
			} else if (("--encode-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				encodeData = xargs(argi).toInt
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
		
		// 1.1. Read header.
		
		val features : Array[String] = SparkUtils.readSmallFile(sc, headerFile)
										// .first.split("\t")
		val featureTypes : Array[Int] = features.map(field => {if (field.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		
		// 1.2 Read data and index it.
		val rawSamples : RDD[String] = sc.textFile(dataFile)
			
		// 1.3. Index categorical features/fields and save indexes.
		println("\n  Generating/reading indexes.\n")
		var indexes :  Array[Map[String,Int]] = null
		if (generateIndexes == 1) {
			indexes = SparkIndexing.generateIndexes(rawSamples, featureTypes)
			SparkIndexing.saveIndexes(sc, indexesDir, features, indexes)
		} else {
			indexes = SparkIndexing.readIndexes(sc, indexesDir, features)
		}
		
		// 1.4. Encode data and save indexed data.
		if (encodeData == 1) {
			println("\n  Encoding data.\n")
			val samples : RDD[Array[Double]] = SparkIndexing.indexRawData(rawSamples, featureTypes, indexes)
			SparkIndexing.saveIndexedData(indexedDataFile, samples, featureTypes)
		}
		
	}

}
