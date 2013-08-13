package boosted_trees

import java.io.File
import java.io.PrintWriter

import scala.io.Source


object DataSampler {
	
	def main(args : Array[String]) : Unit = {
		
		// 0.0. Default parameters.
		
		var dataFile : String = DefaultParameters.dataFile
		var sampleDataFile : String = DefaultParameters.sampleDataFile
		var sampleRate : Double = 0.01
		
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
		
		
		// 1. Count lines.
		var numLines : Long = 0
		var linesIter = Source.fromFile(new File(dataFile)).getLines
		while (linesIter.hasNext) {
			numLines += 1
			linesIter.next
		}
		
		
		// 2. Generate sample line ids.
		
		val numSampleLines : Long = (sampleRate * numLines).toLong
		val sampleLineIds = Utils.sampleWithoutReplacement(numLines, numSampleLines)
		
		
		// 3. Filter sample lines and save them.
		
		Utils.createParentDirs(sampleDataFile)
		val sampleLinesPrintWriter : PrintWriter = new PrintWriter(new File(sampleDataFile))
		linesIter = Source.fromFile(new File(dataFile)).getLines
		var lineId : Long = 0
		while (linesIter.hasNext) {
			val line = linesIter.next
			if (sampleLineIds.contains(lineId)) {
				sampleLinesPrintWriter.println(line)
			}
			lineId += 1
		}
		sampleLinesPrintWriter.close
		
	}

}