package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source
import scala.collection.mutable.MutableList


object GBRTErrorAnalyzer {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		
		var headerFile : String = DefaultParameters.headerFile
		var dataFile : String = DefaultParameters.testDataFile
		var indexesDir : String = DefaultParameters.indexesDir
		var indexedDataFile : String = DefaultParameters.indexedTestDataFile
		var modelDir : String = DefaultParameters.forestModelDir
		var errorDir : String = modelDir
		var binaryMode : Int = 0
		var threshold : Double = 0.5
		var maxNumSummarySamples : Int = 100000
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
			} else if (("--error-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				errorDir = xargs(argi)
			} else if (("--binary-mode".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				binaryMode = xargs(argi).toInt
			} else if (("--threshold".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				threshold = xargs(argi).toDouble
			} else if (("--max-num-summary-samples".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				maxNumSummarySamples = xargs(argi).toInt
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
		
		
		// 1. Read model, read test data and index.
				
		println("\n  Reading model, reading test data and indexing it.\n")
		
		// 1.1. Read forest model.
		val rootNodes : Array[Node] = GBRT.readForest(modelDir + "/nodes/")
		
		// 1.2. Read header.
		val features : Array[String] = Source.fromFile(new File(headerFile)).getLines.toArray
											// .first.split("\t")
		val featureTypes : Array[Int] = features.map(feature => {if (feature.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		
		val numFeatures : Int = featureTypes.length
				
		// 1.3. Read data.
		
		var testSamples : List[Array[Double]] = Nil
		
		if (useIndexedData == 0) {
			// Read indexes for categorical features.
			val indexes : Array[Map[String, Int]] = Indexing.readIndexes(indexesDir, features)
			
			// Read data and encode categorical features.
			testSamples = Indexing.indexRawData(
					Source.fromFile(new File(dataFile)).getLines, featureTypes, indexes)
					
			// Save indexed data.
			if (saveIndexedData == 1) {
				Indexing.saveIndexedData(indexedDataFile, testSamples, featureTypes)
			}
		} else {
			// Use indexed data.
			testSamples = Indexing.readIndexedData(indexedDataFile)
		}
		
		
		// 2. Predict using forest model and evaluate error.
		
		println("\n  Predicting on test data and analyzing error.\n")
		
		val errorStats : (Long, Double, Double) = testSamples.par.map(testSample => {
					val predicted : Double = GBRT.predict(testSample, rootNodes)
					(1L, (predicted - testSample(0)) * (predicted - testSample(0)),
						math.abs(predicted - testSample(0)))
				}).reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
		val trivialResponse = rootNodes.map(_.response).reduce(_ + _)
		val trivialErrorStats : (Long, Double, Double) = testSamples.map(testSample => {
					(1L, (trivialResponse - testSample(0)) * (trivialResponse - testSample(0)),
						math.abs(trivialResponse - testSample(0)))
				}).reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
		
		var binaryErrorStats : (Long, Long, Long, Long) = null
		if (binaryMode == 1) {
			binaryErrorStats = testSamples.map(testSample => {
					val predicted : Int = GBRT.binaryPredict(testSample, rootNodes, threshold)
					val stats : (Long, Long, Long, Long) = testSample(0).toInt match {
						case 0 => (1L, predicted, 0L, 0)
						case 1 => (0L, 0, 1L, 1 - predicted)
					}
					stats
				}).reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3,
						stats1._4 + stats2._4))
		}
		
		// 2.1 Samples for scatter plot.
		
		val numSummarySamples : Int = Math.min(maxNumSummarySamples, testSamples.length)
		val summarySampleIds : Set[Long] =
				Utils.sampleWithoutReplacement(testSamples.length, numSummarySamples)
		val summarySamples : List[Array[Double]] = testSamples.par.zipWithIndex.
				filter(sampleId => summarySampleIds.contains(sampleId._2)).map(_._1).toList
		val predictedVsActual : List[(Double, Double)] = summarySamples.map(testSample => 
					(GBRT.predict(testSample, rootNodes), testSample(0)))
					
					
		// 3. Calculate AUC for binary case.
		
		var roc : Array[(Double, Double, Double)] = null
		var auc : Double = 0
		if (binaryMode == 1) {
			val scoresLabels : List[(Double, Int)] = predictedVsActual.map(x => (x._1, x._2.toInt))
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
			lines += "A = " + (tn + tp) + "/" + (tn + tp + fn + fp) + " = " +
					"%.5f".format((tn + tp).toDouble / (tn + tp + fn + fp))
			lines += "AUC = " + "%.5f".format(auc)
		}
		lines += "RMSE = " + "%.5f".format(math.sqrt(errorStats._2 / errorStats._1))
		lines += "MAE = " + "%.5f".format(errorStats._3 / errorStats._1)
		lines += "Trivial response = " + "%.5f".format(trivialResponse)
		lines += "Trivial RMSE = " + "%.5f".format(math.sqrt(trivialErrorStats._2 / trivialErrorStats._1))
		lines += "Trivial MAE = " + "%.5f".format(trivialErrorStats._3 / trivialErrorStats._1)
		(new File(errorDir)).mkdirs
		var printWriter : PrintWriter = new PrintWriter(new File(errorDir + "/error.txt"))
		for (line <- lines) {
			printWriter.println(line)
		}
		printWriter.close
		printWriter = new PrintWriter(new File(errorDir + "/scatter_plot.txt"))
		printWriter.println(predictedVsActual.map(x => "%.5f".format(x._1) + "\t" +
					"%.5f".format(x._2)).mkString("\n"))
		printWriter.close
		if (binaryMode == 1) {
			printWriter = new PrintWriter(new File(errorDir + "/roc.txt"))
			printWriter.println(roc.map(x => "%.5f".format(x._1) + "\t" +
						"%.5f".format(x._2) + "\t" +
						"%.5f".format(x._3)).mkString("\n"))
			printWriter.close
		}
		
	}

}