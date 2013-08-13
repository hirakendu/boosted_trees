package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source


object DataIndexer {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var headerFile : String = DefaultParameters.headerFile
		var dataFile : String = DefaultParameters.trainDataFile
		var indexesDir : String = DefaultParameters.indexesDir
		var indexedDataFile : String = DefaultParameters.indexedTrainDataFile
		var saveIndexedData : Int = 1

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
		val featureTypes : Array[Int] = features.map(feature => {if (feature.endsWith("$")) 1 else 0})
			// 0 -> continuous, 1 -> discrete
		
		// 1.2 Read data.
		// val rawSamples : List[String] = Source.fromFile(new File(dataFile)).getLines.toArray.toList
		
		// 1.3. Index categorical features/fields and save indexes.
		val indexes :  Array[Map[String,Int]] =
			Indexing.generateIndexes(Source.fromFile(new File(dataFile)).getLines, featureTypes)
		Indexing.saveIndexes(indexesDir, features, indexes)
		
		// 1.4. Encode data and save indexed data.
		if (saveIndexedData == 1) {
			val samples : List[Array[Double]] =
					Indexing.indexRawData(Source.fromFile(new File(dataFile)).getLines, featureTypes, indexes)
			Indexing.saveIndexedData(indexedDataFile, samples, featureTypes)
		}
		
	}
	

}