package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object SparkDataSampler {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var sparkMaster : String = SparkDefaultParameters.sparkMaster
		var sparkHome : String = SparkDefaultParameters.sparkHome
		val sparkAppName : String = SparkDefaultParameters.sparkAppName
		var sparkAppJars : String = SparkDefaultParameters.sparkAppJars
		var dataFile : String = SparkDefaultParameters.dataFile
		var sampleDataFile : String = SparkDefaultParameters.sampleDataFile
		var sampleRate : Double = 0.01
		
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
			} else if (("--sample-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				sampleDataFile = xargs(argi)
			} else if (("--sample-rate".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				sampleRate = xargs(argi).toDouble
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
		
		
		// 1. Generate sample lines.

		val lines : RDD[String] = sc.textFile(dataFile)
		val sampleLines : RDD[String] = lines.sample(false, sampleRate, 42)
		
		
		// 2. Save.
		
		sampleLines.saveAsTextFile(sampleDataFile)
		
	}

}
