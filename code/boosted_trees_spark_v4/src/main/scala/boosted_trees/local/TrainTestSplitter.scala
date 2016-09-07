package boosted_trees.local

import java.io.File
import java.io.PrintWriter

import scala.io.Source

import boosted_trees.Utils


object TrainTestSplitter {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    var dataFile: String = DefaultParameters.dataFile
    var trainDataFile: String = DefaultParameters.trainDataFile
    var testDataFile: String = DefaultParameters.testDataFile
    var trainFraction: Double = 0.8

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
      } else if (("--train-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        trainDataFile = xargs(argi)
      } else if (("--test-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        testDataFile = xargs(argi)
      } else if (("--train-fraction".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        trainFraction = xargs(argi).toDouble
      } else {
        println("\n  Error parsing argument \"" + xargs(argi) +
            "\".\n")
        return
      }
      argi += 1
    }


    // 1. Count samples.

    var numSamples: Long = 0
    var samplesIter = Source.fromFile(new File(dataFile)).getLines
    while (samplesIter.hasNext) {
      numSamples += 1
      samplesIter.next
    }


    // 2. Generate test sample ids.

    val numTestSamples: Long = ((1 - trainFraction) * numSamples).toLong
    val testSampleIds = Utils.sampleWithoutReplacement(numSamples, numTestSamples)


    // 3. Filter training and test samples and save them.

    Utils.createParentDirs(trainDataFile)
    Utils.createParentDirs(testDataFile)
    val trainSamplesPrintWriter: PrintWriter = new PrintWriter(new File(trainDataFile))
    val testSamplesPrintWriter: PrintWriter = new PrintWriter(new File(testDataFile))
    samplesIter = Source.fromFile(new File(dataFile)).getLines
    var sampleId: Long = 0
    while (samplesIter.hasNext) {
      if (testSampleIds.contains(sampleId)) {
        testSamplesPrintWriter.println(samplesIter.next)
      } else {
        trainSamplesPrintWriter.println(samplesIter.next)
      }
      sampleId += 1
    }
    trainSamplesPrintWriter.close
    testSamplesPrintWriter.close

  }

}
