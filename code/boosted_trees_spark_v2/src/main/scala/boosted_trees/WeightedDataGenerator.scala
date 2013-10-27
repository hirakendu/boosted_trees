package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source
import scala.collection.mutable.MutableList


object WeightedDataGenerator {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var headerFile : String = DefaultParameters.headerFile
		var weightedDataHeaderFile : String = DefaultParameters.weightedDataHeaderFile
		var weightStepsFile : String = DefaultParameters.weightStepsFile
		var dataFile : String = DefaultParameters.indexedTrainDataFile
		var weightedDataFile : String = DefaultParameters.weightedDataFile
	
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
		
		
		// 1. Read data and add weight column.
		
		// 1.1. Read header.
		val features : Array[String] =  Source.fromFile(new File(headerFile)).
				getLines.toArray
		
		// 1.2. Read weight steps.
		val weightStepsLines : List[String] = Source.fromFile(new File(weightStepsFile)).getLines.toList
		val weightSteps : MutableList[(Double, Double)] = MutableList()
		for (s <- 0 to (weightStepsLines.length  - 3) / 2) {
			weightSteps += ((weightStepsLines(2 * s + 1).toDouble, weightStepsLines(2 * s).toDouble))
		}
		weightSteps += ((9000, weightStepsLines(weightStepsLines.length - 1).toDouble))
		
		// 1.3. Read data.
		val samples : List[String] = Source.fromFile(new File(dataFile)).getLines.toArray.toList
		
		// 1.4. Generate weighted samples.
		val weightedSamples : List[String] = samples.par.map(sample => {
				val response : Double = sample.split("\t")(0).toDouble
				var sampleWeight : Double = weightSteps(weightSteps.length - 1)._2
				for (s <- 0 to weightSteps.length - 2) {
					if (response < weightSteps(s)._1) {
						sampleWeight = weightSteps(s)._2
					}
				}
				sample + "\t" + sampleWeight
			}).toList
		
		// 1.5. Save weighted data and header.
		var printWriter : PrintWriter = new PrintWriter(new File(weightedDataHeaderFile))
		printWriter.println(features.mkString("\n"))
		printWriter.println("sample_weight")
		printWriter.close
		printWriter = new PrintWriter(new File(weightedDataFile))
		for (line <- weightedSamples) {
			printWriter.println(line)
		}
		printWriter.close
		
	}

}
