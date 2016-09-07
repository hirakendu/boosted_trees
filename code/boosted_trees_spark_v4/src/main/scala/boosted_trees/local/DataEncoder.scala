package boosted_trees.local

import java.io.File

import scala.io.Source

import boosted_trees.LabeledPoint


object DataEncoder {

  def main(args: Array[String]): Unit = {

    // 0.0. Default parameters.

    var headerFile: String = DefaultParameters.headerFile
    var dataFile: String = DefaultParameters.trainDataFile
    var dictsDir: String = DefaultParameters.dictsDir
    var encodedDataFile: String = DefaultParameters.encodedTrainDataFile
    var generateDicts: Int = 1
    var encodeData: Int = 1

    var maxNumQuantiles: Int = 1000

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
      } else if (("--dicts-dir".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        dictsDir = xargs(argi)
      } else if (("--encoded-data-file".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        encodedDataFile = xargs(argi)
      }  else if (("--generate-dicts".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        generateDicts = xargs(argi).toInt
      } else if (("--encode-data".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        encodeData = xargs(argi).toInt
      } else if (("--max-num-quantiles".equals(xargs(argi))) && (argi + 1 < xargs.length)) {
        argi += 1
        maxNumQuantiles = xargs(argi).toInt
      } else {
        println("\n  Error parsing argument \"" + xargs(argi) +
            "\".\n")
        return
      }
      argi += 1
    }


    // 1. Read input data and index it.

    // 1.1. Read header.

    val features: Array[String] = Source.fromFile(new File(headerFile)).getLines.toArray.drop(1)
                    // .first.split("\t").drop(1)
    val numFeatures: Int = features.length
    val featureTypes: Array[Int] = features.map(feature => {
        if (feature.endsWith("$")) 1 else if (feature.endsWith("#")) -1 else 0})
        // 0 -> continuous, 1 -> discrete, -1 ignored

    // 1.2 Read data.

    val rawSamples: Array[String] = Source.fromFile(new File(dataFile)).getLines.toArray

    // 1.3. Index categorical features, find quantiles for continuous features
    //      and encode data.

    println("\n  Generating/reading indexes.\n")
    var indexes:  Array[Map[String,Int]] = null
    var quantilesForFeatures: Array[Array[Double]] = null
    var samples: Array[LabeledPoint] = null
    if (generateDicts == 1) {
      indexes = DataEncoding.generateIndexes(rawSamples, featureTypes)
      samples = DataEncoding.encodeRawData(rawSamples, featureTypes, indexes)
      quantilesForFeatures = DataEncoding.findQuantilesForFeatures(
          samples, featureTypes, maxNumQuantiles)
      DataEncoding.saveIndexes(dictsDir, features, indexes)
      DataEncoding.saveQuantiles(dictsDir, features, quantilesForFeatures)
    } else {
      indexes = DataEncoding.readIndexes(dictsDir, features)
      samples = DataEncoding.encodeRawData(rawSamples, featureTypes, indexes)
    }

    // 1.4. Save encoded data.

    if (encodeData == 1) {
      println("\n  Saving encoded data.\n")
      DataEncoding.saveEncodedData(encodedDataFile, samples, featureTypes)
    }

  }

}
