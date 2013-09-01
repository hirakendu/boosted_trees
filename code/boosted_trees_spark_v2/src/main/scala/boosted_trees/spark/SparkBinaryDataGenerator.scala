package boosted_trees.spark

import spark.RDD
import spark.SparkContext
import spark.SparkContext._

import scala.collection.mutable.MutableList


object SparkBinaryDataGenerator {
	
	def main(args : Array[String]) : Unit = {

		// 0.0. Default parameters.
		var sparkMaster : String = SparkDefaultParameters.sparkMaster
		var sparkHome : String = SparkDefaultParameters.sparkHome
		var sparkAppJars : String = SparkDefaultParameters.sparkAppJars
		var dataFile : String = SparkDefaultParameters.trainDataFile
		var binaryDataFile : String = SparkDefaultParameters.binaryDataFile
		var threshold : Double = 0.5

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
			} else if (("--data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				dataFile = xargs(argi)
			} else if (("--binary-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				binaryDataFile = xargs(argi)
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
		
		// 0.2. Create Spark context.
		var sparkAppJarsSeq : Seq[String] = Nil
		if (sparkAppJars != null) {
			sparkAppJarsSeq = sparkAppJars.split(",").toSeq
		}
		val sc : SparkContext = new SparkContext(sparkMaster, "Spark Regression Tree",
				sparkHome, sparkAppJarsSeq)
		
		
		// 1. Threshold data into binary.
		
		// 1.1. Read data.
		val samples : RDD[String] = sc.textFile(dataFile)
		
		// 1.2. Generate binary responses for samples.
		val binarySamples : RDD[String] = samples.map(sample => {
				val values : Array[String] =  sample.split("\t")
				val response : Double = values(0).toDouble
				var b : Int = 0
				if (response >= threshold) {
					b = 1
				}
				b + "\t" + values.drop(1).mkString("\t")
			})
		
		// 1.3. Save binary data.
		binarySamples.saveAsTextFile(binaryDataFile)
		
	}


}