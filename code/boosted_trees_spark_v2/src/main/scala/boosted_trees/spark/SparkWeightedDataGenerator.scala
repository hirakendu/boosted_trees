package boosted_trees.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import scala.collection.mutable.MutableList


object SparkWeightedDataGenerator {
	
	def main(args : Array[String]) : Unit = {

		// 0.0. Default parameters.
		var sparkMaster : String = SparkDefaultParameters.sparkMaster
		var sparkHome : String = SparkDefaultParameters.sparkHome
		val sparkAppName : String = SparkDefaultParameters.sparkAppName
		var sparkAppJars : String = SparkDefaultParameters.sparkAppJars
		var headerFile : String = SparkDefaultParameters.headerFile
		var weightedDataHeaderFile : String = SparkDefaultParameters.weightedDataHeaderFile
		var weightStepsFile : String = SparkDefaultParameters.weightStepsFile
		var dataFile : String = SparkDefaultParameters.indexedTrainDataFile
		var weightedDataFile : String = SparkDefaultParameters.weightedDataFile

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
			} else if (("--weighted-data-header-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				weightedDataHeaderFile = xargs(argi)
			} else if (("--weight-steps-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				weightStepsFile = xargs(argi)
			} else if (("--data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				dataFile = xargs(argi)
			} else if (("--weighted-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				weightedDataFile = xargs(argi)
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
		
		
		// 1. Read data and add weight column.
		
		println("\n  Reading data and adding weight column.\n")
		
		// 1.1. Read header.
		
		val features : Array[String] = SparkUtils.readSmallFile(sc, headerFile)
										// .first.split("\t")
		
		// 1.2. Read weight steps.
		val weightStepsLines : Array[String] = SparkUtils.readSmallFile(sc, weightStepsFile)
		val weightSteps : MutableList[(Double, Double)] = MutableList()
		for (s <- 0 to (weightStepsLines.length  - 3) / 2) {
			weightSteps += ((weightStepsLines(2 * s + 1).toDouble, weightStepsLines(2 * s).toDouble))
		}
		weightSteps += ((9000, weightStepsLines(weightStepsLines.length - 1).toDouble))
		
		// 1.3. Read data.
		val samples : RDD[String] = sc.textFile(dataFile)
		
		// 1.4. Generate weighted samples.
		val weightedSamples : RDD[String] = samples.map(sample => {
				val response : Double = sample.split("\t")(0).toDouble
				var sampleWeight : Double = weightSteps(weightSteps.length - 1)._2
				for (s <- 0 to weightSteps.length - 2) {
					if (response < weightSteps(s)._1) {
						sampleWeight = weightSteps(s)._2
					}
				}
				sample + "\t" + sampleWeight
			})
		
		// 1.5. Save weighted data and header.
		weightedSamples.saveAsTextFile(weightedDataFile)
		sc.parallelize(features.toArray ++ Array("sample_weight"), 1).
			saveAsTextFile(weightedDataHeaderFile)
		
	}

}
