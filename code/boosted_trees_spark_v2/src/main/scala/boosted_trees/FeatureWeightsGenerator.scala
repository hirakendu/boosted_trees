package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source
import scala.collection.mutable.{Set => MuSet}


object FeatureWeightsGenerator {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var headerFile : String = DefaultParameters.headerFile
		var includedFeaturesFile : String = DefaultParameters.includedFeaturesFile
		var excludedFeaturesFile : String = DefaultParameters.excludedFeaturesFile
		var featureWeightsFile : String = DefaultParameters.featureWeightsFile
		
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
			} else if (("--included-features-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				includedFeaturesFile = xargs(argi)
			} else if (("--excluded-features-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				excludedFeaturesFile = xargs(argi)
			} else if (("--feature-weights-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
				argi += 1
				featureWeightsFile = xargs(argi)
			} else {
				println("\n  Error parsing argument \"" + xargs(argi) +
						"\".\n")
				return
			}
			argi += 1
		}
		
		
		// 1. Read list of all features, and features to include and exclude.
		val features : Array[String] =  Source.fromFile(new File(headerFile)).
				getLines.toArray
		
		val numFeatures : Int = features.length
		
		var includedFeatures : Set[String] = features.toSet
		if (!includedFeaturesFile.equals("")) {
			includedFeatures = Source.fromFile(new File(includedFeaturesFile)).
					getLines.toSet
		}
		
		
		var excludedFeatures : Set[String] = Set()
		if (!excludedFeaturesFile.equals("")) {
			excludedFeatures = Source.fromFile(new File(excludedFeaturesFile)).
					getLines.toSet
		}
		
		// 2. Generate feature weights and save.
		
		val featureWeights : Array[Double] = new Array(numFeatures)
		featureWeights(0) = 1
		for (j <- 1 to numFeatures - 1) {
			if (includedFeatures.contains(features(j)) &&
					!excludedFeatures.contains(features(j))) {
				featureWeights(j) = 1
			}
		}
		
		Utils.createParentDirs(featureWeightsFile)
		val printWriter : PrintWriter = new PrintWriter(new File(featureWeightsFile))
		printWriter.println(featureWeights.mkString("\n"))
		printWriter.close
		
	}
	
}