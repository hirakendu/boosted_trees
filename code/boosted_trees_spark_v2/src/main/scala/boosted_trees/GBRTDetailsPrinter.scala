package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source


object GBRTDetailsPrinter {
	
		def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		
		var headerFile : String = DefaultParameters.headerFile
		var indexesDir : String = DefaultParameters.indexesDir
		var modelDir : String = DefaultParameters.forestModelDir
		
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
			} else if (("--indexes-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				indexesDir = xargs(argi)
			} else if (("--model-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				modelDir = xargs(argi)
			} else {
				println("\n  Error parsing argument \"" + xargs(argi) +
						"\".\n")
				return
			}
			argi += 1
		}
		
		
		// 1. Read forest model, header and indexes.
		
		// 1.1. Read forest model.
		val rootNodes : Array[Node] = GBRT.readForest(modelDir + "/nodes/")
		
		// 1.2. Read header.
		val features : Array[String] = Source.fromFile(new File(headerFile)).getLines.toArray
											// .first.split("\t")
		val featureTypes : Array[Int] = features.map(feature => {if (feature.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		
		val numFeatures : Int = featureTypes.length
				
		// 1.3. Read indexes.
		
		val indexes : Array[Map[String, Int]] = Indexing.readIndexes(indexesDir, features)
		
		
		// 2. Print forest details.
		
		GBRT.printForest(modelDir + "/trees_details/", rootNodes, features, indexes)
		
		val featureGains : Array[Double] = GBRT.evaluateFeatureGains(rootNodes, numFeatures)
		var maxGain : Double = featureGains.max
		val featureImportances : Array[String] = features.
			zip(featureGains.map(_ * 100 / maxGain)).
			drop(1).  // Drop the label feature
			toArray.sortWith(_._2 > _._2).
			map(featureImportance => featureImportance._1 + "\t" + "%.1f".format(featureImportance._2))
		var printWriter : PrintWriter = new PrintWriter(new File(modelDir + "/feature_importances.txt"))
		printWriter.println(featureImportances.mkString("\n"))
		printWriter.close
		
		
		val featureSubsetGains : Array[(Set[Int], Double)] =
				GBRT.evaluateFeatureSubsetGains(rootNodes)
		maxGain = featureSubsetGains.map(_._2).max
		val featureSubsetImportances : Array[String] =
			featureSubsetGains.map(x => x._1.map(features(_)).mkString(",") +
					"\t" + "%.1f".format(x._2 * 100 / maxGain))
		printWriter = new PrintWriter(new File(modelDir + "/feature_subset_importances.txt"))
		printWriter.println(featureSubsetImportances.mkString("\n"))
		printWriter.close
		
	}

}
