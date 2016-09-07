package boosted_trees.local

import java.io.File
import java.io.PrintWriter

import scala.io.Source

import boosted_trees.Utils


object DataSampler {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    var dataFile: String = DefaultParameters.dataFile
    var sampleDataFile: String = DefaultParameters.workDir + "/sample_data.txt"
    var sampleFraction: Double = 0.01

    // 0.1. Read parameters.

    var xargs: Array[String] = args
    if (args.length == 1) {
      xargs = args(0).split("\\^")
    }
    var argi: Int = 0
    while (argi < xargs.length) {
      if (("--data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        dataFile = xargs(argi)
      } else if (("--sample-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        sampleDataFile = xargs(argi)
      } else if (("--sample-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        sampleFraction = xargs(argi).toDouble
      } else {
        println("\n  Error parsing argument \"" + xargs(argi) +
            "\".\n")
        return
      }
      argi += 1
    }


    // 1. Count lines.

    val numLines: Long = Source.fromFile(new File(dataFile)).getLines.toArray.length


    // 2. Generate sample line ids.

    val numSampleLines: Long = (sampleFraction * numLines).toLong
    val sampleLineIds: Set[Long] = Utils.sampleWithoutReplacement(numLines, numSampleLines)


    // 3. Filter sample lines and save them.

    Utils.createParentDirs(sampleDataFile)
    val printWriter: PrintWriter = new PrintWriter(new File(sampleDataFile))
    val lines: Array[String] = Source.fromFile(new File(dataFile)).getLines.toArray
    printWriter.println(Source.fromFile(new File(dataFile)).getLines.toArray.
        zipWithIndex.filter(x => sampleLineIds.contains(x._2)).mkString("\n"))
    printWriter.close

  }

}
