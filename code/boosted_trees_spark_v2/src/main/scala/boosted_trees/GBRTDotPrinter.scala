package boosted_trees

import java.io.File

import scala.io.Source


object GBRTDotPrinter {
	
		def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		
		var headerFile : String = DefaultParameters.headerFile
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
		
		
		// 2. Print forest details.
		
		GBRT.printForestDot(modelDir + "/trees_dot/", rootNodes, features)
		
	}

}
