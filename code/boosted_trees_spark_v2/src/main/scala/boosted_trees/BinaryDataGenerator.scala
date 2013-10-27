package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source


object BinaryDataGenerator {

	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		var dataFile : String = DefaultParameters.dataFile
		var binaryDataFile : String = DefaultParameters.binaryDataFile
		var threshold : Double = 0.5
	
		// 0.1. Read parameters.
		
		var xargs : Array[String] = args
		if (args.length == 1) {
			xargs = args(0).split("\\^")
		}
		var argi : Int = 0
		while (argi < xargs.length) {
			if (("--data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
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
		
		
		// 1. Threshold data into binary.
		
		// 1.1. Read data.
		val samples : List[String] = Source.fromFile(new File(dataFile)).getLines.toArray.toList
		
		// 1.2. Generate binary responses for samples.
		val binarySamples : List[String] = samples.par.map(sample => {
				val values : Array[String] =  sample.split("\t")
				val response : Double = values(0).toDouble
				var b : Int = 0
				if (response >= threshold) {
					b = 1
				}
				b + "\t" + values.drop(1).mkString("\t")
			}).toList
		
		// 1.3. Save binary data.
		val printWriter : PrintWriter = new PrintWriter(new File(binaryDataFile))
		for (line <- binarySamples) {
			printWriter.println(line)
		}
		printWriter.close
		
	}

}
