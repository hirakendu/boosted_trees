package boosted_trees.spark

import scala.collection.mutable.{Set => MuSet}
import scala.collection.mutable.{Map => MuMap}
import scala.collection.mutable.MutableList
import scala.collection.mutable.Stack
import scala.collection.parallel.immutable.ParSeq

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import boosted_trees.Node
import boosted_trees.RegressionTree
import boosted_trees.Utils


object SparkRegressionTreeErrorAnalyzer {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var sparkMaster : String = SparkDefaultParameters.sparkMaster
		var sparkHome : String = SparkDefaultParameters.sparkHome
		val sparkAppName : String = SparkDefaultParameters.sparkAppName
		var sparkAppJars : String = SparkDefaultParameters.sparkAppJars
		var headerFile : String = SparkDefaultParameters.headerFile
		var dataFile : String = SparkDefaultParameters.testDataFile
		var indexesDir : String = SparkDefaultParameters.indexesDir
		var indexedDataFile : String = SparkDefaultParameters.indexedTestDataFile
		var modelDir : String = SparkDefaultParameters.treeModelDir
		var errorFile : String = modelDir + "/error.txt"
		var rocFile : String = modelDir + "/roc.txt"
		var binaryMode : Int = 0
		var threshold : Double = 0.5
		var maxNumRocSamples : Int = 100000
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
			} else if (("--error-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				errorFile = xargs(argi)
			} else if (("--roc-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				rocFile = xargs(argi)
			} else if (("--binary-mode".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				binaryMode = xargs(argi).toInt
			} else if (("--threshold".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				threshold = xargs(argi).toDouble
			} else if (("--max-num-roc-samples".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				maxNumRocSamples = xargs(argi).toInt
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
		val sc : SparkContext = new SparkContext(sparkMaster, sparkAppName,
				sparkHome, sparkAppJarsSeq)
		
		
		// 1. Read model, read test data and index.
				
		println("\n  Reading model, reading test data and indexing it.\n")
		
		// 1.1. Read tree model.
		val rootNode : Node = SparkRegressionTree.readTree(sc, modelDir + "/nodes.txt")
		
		// 1.2. Read header.
		val features : Array[String] = SparkUtils.readSmallFile(sc, headerFile)
										// .first.split("\t")
		val featureTypes : Array[Int] = features.map(feature => {if (feature.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		
		val numFeatures : Int = featureTypes.length
		
		// 1.3. Read data.
		
		var testSamples : RDD[Array[Double]] = null
		
		if (useIndexedData == 0) {
			// Read indexes for categorical features.
			val indexes : Array[Map[String, Int]] = SparkIndexing.readIndexes(sc, indexesDir, features)
			
			// Read data and encode categorical features.
			testSamples = SparkIndexing.indexRawData(sc, dataFile, featureTypes, indexes)
			
			// Save indexed data.
			if (saveIndexedData == 1) {
				SparkIndexing.saveIndexedData(indexedDataFile, testSamples, featureTypes)
			}
		} else {
			// Use indexed data.
			testSamples = SparkIndexing.readIndexedData(sc, indexedDataFile)
		}
		
		if (cacheIndexedData == 1) {
			testSamples.cache
		}
		
		
		// 2. Predict using tree model and evaluate error.
		
		println("\n  Predicting on test data and analyzing error.\n")
		
		val errorStats : (Long, Double, Double) = testSamples.map(testSample => {
					val predicted : Double = RegressionTree.predict(testSample, rootNode)
					(1L, (predicted - testSample(0)) * (predicted - testSample(0)),
						math.abs(predicted - testSample(0)))
				}).reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
		val trivialResponse = rootNode.response
		val trivialErrorStats : (Long, Double, Double) = testSamples.map(testSample => {
					(1L, (trivialResponse - testSample(0)) * (trivialResponse - testSample(0)),
						math.abs(trivialResponse - testSample(0)))
				}).reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
		
		var binaryErrorStats : (Long, Long, Long, Long) = null
		if (binaryMode == 1) {
			binaryErrorStats = testSamples.map(testSample => {
					val predicted : Int = RegressionTree.binaryPredict(testSample, rootNode, threshold)
					val stats : (Long, Long, Long, Long) = testSample(0).toInt match {
						case 0 => (1L, predicted, 0L, 0)
						case 1 => (0L, 0, 1L, 1 - predicted)
					}
					stats
				}).reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3,
						stats1._4 + stats2._4))
		}
		
		
		// 3. Calculate AUC for binary case.
		
		var roc : Array[(Double, Double, Double)] = null
		var auc : Double = 0
		if (binaryMode == 1) {
			val numRocSamples : Int = Math.min(maxNumRocSamples, testSamples.count).toInt
			val rocSamples : List[Array[Double]] = testSamples.takeSample(false, numRocSamples, 42).toList
			val scoresLabels : List[(Double, Int)] = rocSamples.map(testSample => 
					(RegressionTree.predict(testSample, rootNode), testSample(0).toInt))
			val rocAuc = Utils.findRocAuc(scoresLabels)
			roc = rocAuc._1
			auc = rocAuc._2
		}
		
		
		// 4. Save error statistics.
		
		println("\n  Saving error statistics.\n")
		
		val lines : MutableList[String] = MutableList()
		if (binaryMode == 1) {
			val fp : Long =  binaryErrorStats._2
			val fn : Long =  binaryErrorStats._4
			val tp : Long =  binaryErrorStats._3 - fn
			val tn : Long =  binaryErrorStats._1 - fp
			lines += "TPR = Recall = " + tp + "/" + (tp + fn) + " = " +
					"%.5f".format(tp.toDouble / (tp + fn))
			lines += "FPR = " + fp + "/" + (tn + fp) + " = " +
					"%.5f".format(fp.toDouble / (tn + fp))
			lines += "Precision = " + tp + "/" + (tp + fp) + " = " +
					"%.5f".format(tp.toDouble / (tp + fp))
			lines += "F1 = " +  "%.5f".format(2 * tp.toDouble / (2 * tp + fn + fp))
			lines += "A = " + "%.5f".format((tn + tp).toDouble / (tn + tp + fn + fp))
			lines += "AUC = " + "%.5f".format(auc)
		}
		lines += "RMSE = " + "%.5f".format(math.sqrt(errorStats._2 / errorStats._1))
		lines += "MAE = " + "%.5f".format(errorStats._3 / errorStats._1)
		lines += "Trivial response = " + "%.5f".format(trivialResponse)
		lines += "Trivial RMSE = " + "%.5f".format(math.sqrt(trivialErrorStats._2 / trivialErrorStats._1))
		lines += "Trivial MAE = " + "%.5f".format(trivialErrorStats._3 / trivialErrorStats._1)
		sc.parallelize(lines, 1).saveAsTextFile(errorFile)
		sc.parallelize(roc.map(x => "%.5f".format(x._1) + "\t" +
						"%.5f".format(x._2) + "\t" + "%.5f".format(x._3)), 1).
						saveAsTextFile(rocFile)
		
	}

}