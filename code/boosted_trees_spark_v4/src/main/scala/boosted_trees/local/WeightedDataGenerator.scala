package boosted_trees.local

import java.io.File
import java.io.PrintWriter

import scala.io.Source


object WeightedDataGenerator {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    var headerFile: String = DefaultParameters.headerFile
    var dataFile: String = DefaultParameters.trainDataFile
    var weightStepsFile: String = DefaultParameters.workDir + "/weight_steps.txt"
    var weightedDataHeaderFile: String = DefaultParameters.workDir + "/weighted_data_header.txt"
    var weightedDataFile: String = DefaultParameters.workDir + "/weighted_train_data.txt"

    // 0.1. Read parameters.

    var xargs: Array[String] = args
    if (args.length == 1) {
      xargs = args(0).split("\\^")
    }
    var argi: Int = 0
    while (argi < xargs.length) {
      if (("--header-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        headerFile = xargs(argi)
      } else if (("--data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        dataFile = xargs(argi)
      } else if (("--weight-steps-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        weightStepsFile = xargs(argi)
      } else if (("--weighted-data-header-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        weightedDataHeaderFile = xargs(argi)
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

    val features: Array[String] =  Source.fromFile(new File(headerFile)).
        getLines.toArray

    // 1.2. Read weight steps.

    val weightStepsLines: Array[String] = Source.fromFile(new File(weightStepsFile)).getLines.toArray
    val weights: Array[Double] = weightStepsLines.zipWithIndex.filter(_._2 % 2 == 0).map(_._1.toDouble)
    val thresholds: Array[Double] = weightStepsLines.zipWithIndex.filter(_._2 % 2 == 1).map(_._1.toDouble)


    // 1.3. Read data.

    val samples: Array[String] = Source.fromFile(new File(dataFile)).getLines.toArray.toArray

    // 1.4. Generate weighted samples.

    val weightedSamples: Array[String] = samples.map(sample => {
        val response: Double = sample.split("\t")(0).toDouble
        var sampleWeight: Double = weights(weights.length - 1)
        for (s <- 0 to thresholds.length - 1) {
          if (response < thresholds(s)) {
            sampleWeight = weights(s)
          }
        }
        sample + "\t" + sampleWeight
      })

    // 1.5. Save weighted data and header.

    var printWriter: PrintWriter = new PrintWriter(new File(weightedDataHeaderFile))
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
