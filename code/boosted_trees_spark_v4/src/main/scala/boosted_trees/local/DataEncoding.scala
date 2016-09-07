package boosted_trees.local

import java.io.File
import java.io.PrintWriter

import scala.io.Source
// import scala.collection.parallel.mutable.ParArray

import boosted_trees.LabeledPoint

import boosted_trees.Utils


object DataEncoding {

  /**
   * Finds quantiles for continuous features.
   */
  def findQuantilesForFeatures(quantileSamples: Array[LabeledPoint],
      featureTypes: Array[Int],
      maxNumQuantiles: Int = 1000): Array[Array[Double]] = {

    val quantilesForFeatures: Array[Array[Double]] = new Array(featureTypes.length)

    val numQuantileSamples: Int = quantileSamples.length
    val numQuantiles: Int = math.min(maxNumQuantiles, numQuantileSamples - 1)

    for (j <- 0 to featureTypes.length - 1) {
      if (featureTypes(j) == 0) {
        val featureValues: Array[Double] = quantileSamples.map(_.features(j).toDouble).sortWith(_ < _)
        val quantilesSet: Array[Double] = new Array(numQuantiles)
        for (q <- 1 to numQuantiles) {
          val id: Int = (q * numQuantileSamples / (numQuantiles + 1.0) - 1).toInt
          quantilesSet(q - 1) = (featureValues(id) + featureValues(id + 1)) / 2.0
        }
        quantilesForFeatures(j) = quantilesSet.toSet.toArray.sortWith(_ < _)
      }
    }

    quantilesForFeatures
  }

  /**
   * Saves quantiles.
   */
  def saveQuantiles(quantilesDir: String, features: Array[String],
      quantilesForFeatures: Array[Array[Double]]): Unit = {
    for (j <- 0 to features.length - 1) {
      if (!features(j).endsWith("$") && !features(j).endsWith("#")) {
        (new File(quantilesDir)).mkdirs
        val printWriter: PrintWriter = new PrintWriter(new File(quantilesDir + "/" +
                features(j) + "_quantiles.txt"))
        printWriter.println(quantilesForFeatures(j).map("%.5g".format(_)).mkString("\n"))
        printWriter.close
      }
    }
  }

  /**
   * Reads quantiles.
   */
  def readQuantiles(quantilesDir: String,
      features: Array[String]): Array[Array[Double]] = {
    val quantilesForFeatures: Array[Array[Double]] = new Array(features.length)
    for (j <- 0 to features.length - 1) {
      if (!features(j).endsWith("$") && !features(j).endsWith("#")) {
        quantilesForFeatures(j) = Source.fromFile(new File(quantilesDir + "/" +
                features(j) + "_quantiles.txt")).
                getLines.toArray.
                map(_.toDouble).sortWith(_ < _)
      }
    }
    quantilesForFeatures
  }

  /**
   * Looks up the quantile bin for a given value among given quantiles.
   */
  def findQuantileBin(quantiles: Array[Double], value: Double): Int = {
    var b1: Int = 0
    var b2: Int = quantiles.length // Last bin.
    while (b1 < b2) {
      val b3 = (b1 + b2) / 2
      if (value < quantiles(b3)) {
        b2 = b3
      } else {
        b1 = b3 + 1
      }
    }
    b1
  }

  /**
   * Generates indexes/dictionaries for categorical features in a dataset.
   */
  def generateIndexes(rawSamples: Array[String], featureTypes: Array[Int]):
    Array[Map[String, Int]] = {

    // 1. Find the set of unique values, i.e., dictionary
    //    for each categorical feature.
    val rawValuesForFeatures: Array[Set[String]] = new Array(featureTypes.length)

    val rawSamplesSplit: Array[Array[String]] = rawSamples.map(_.split("\t", -1).drop(1))
    for (j <- 0 to featureTypes.length - 1) {
      if (featureTypes(j) == 1) {
        rawValuesForFeatures(j) = rawSamplesSplit.map(_(j)).toSet
      }
    }

    // 2. Index unique values of each categorical feature.
    val indexes: Array[Map[String, Int]] = new Array(featureTypes.length)
    // ParArray(Range(0, featureTypes.length): _*).foreach(j => {
    for (j <- 0 to featureTypes.length - 1) {
      if (featureTypes(j) == 1) {
        indexes(j) = Map(rawValuesForFeatures(j).toArray.zipWithIndex: _*)
      }
    }
    // })
    indexes
  }

  /*
   * Saves indexes.
   */
  def saveIndexes(indexesDir: String, features: Array[String],
      indexes: Array[Map[String, Int]]): Unit = {
    for (j <- 0 to features.length - 1) {
      if (features(j).endsWith("$")) {
        (new File(indexesDir)).mkdirs
        val printWriter: PrintWriter = new PrintWriter(new File(indexesDir + "/" +
                features(j).replace("$", "") + "_index.txt"))
        printWriter.println(indexes(j).toArray.sortWith(_._2 < _._2).
            map(valueId => valueId._1.toString + "\t" + valueId._2).mkString("\n"))
        printWriter.close
      }
    }
  }

  /*
   * Reads indexes.
   */
  def readIndexes(indexesDir: String, features: Array[String]):
    Array[Map[String, Int]] = {

    val indexes: Array[Map[String, Int]] = new Array(features.length)
    for (j <- 0 to features.length - 1) {
      if (features(j).endsWith("$")) {
        indexes(j) = Source.fromFile(new File(indexesDir + "/" +
                features(j).replace("$", "") + "_index.txt")).
                getLines.toArray.
                map(kv => {
                  val kvArray: Array[String] = kv.split("\t", -1)
                  (kvArray(0), kvArray(1).toInt)
                }).toMap
      }
    }
    indexes
  }

  /**
   * Encodes categorical features in a raw sample using given indexes.
   */
  def encodeRawSample(rawSample: String, featureTypes: Array[Int],
      indexes: Array[Map[String, Int]]): LabeledPoint = {

    val rawValues: Array[String] = rawSample.split("\t", -1)
    val label: Double = rawValues(0).toDouble
    val featureValues: Array[Double] = new Array(featureTypes.length)

    // ParArray(Range(0, featureTypes.length): _*).foreach(j => {
    for (j <- 0 to featureTypes.length - 1) {
      if (featureTypes(j) == 0) {
        if ("".equals(rawValues(j + 1))) {
          featureValues(j) = -1.0
        } else {
          featureValues(j) = rawValues(j + 1).toDouble
        }
      } else if (featureTypes(j) == 1) {
        // FIXME: add option to specify default values when
        // the categorical value is not present in index.
        // Currently arbitrarily set to 0.
        featureValues(j) = indexes(j).getOrElse(rawValues(j + 1), 0).toDouble
      } else if (featureTypes(j) == -1) {
        featureValues(j) = -1
      }
    }
    // })

    LabeledPoint(label, featureValues)
  }

  /**
   * Encodes categorical features in data using given indexes.
   */
  def encodeRawData(rawSamples: Array[String], featureTypes: Array[Int],
      indexes: Array[Map[String, Int]]): Array[LabeledPoint] = {
    rawSamples.filter(_.split("\t", -1).length == featureTypes.length + 1).
      map(rawSample => DataEncoding.encodeRawSample(rawSample, featureTypes, indexes))
  }

  /*
   * Saves encoded data.
   */
  def saveEncodedData(encodedDataFile: String, samples: Array[LabeledPoint],
      featureTypes: Array[Int]): Unit = {
    val lines: Array[String] = samples.map(sample => {
        var sampleStr: String = sample.label.toString
        for (j <- 0 to featureTypes.length - 1) {
          sampleStr += "\t"
          if (featureTypes(j) == 0) {
            sampleStr += sample.features(j)
          } else {
            sampleStr += sample.features(j).toInt
          }
        }
        sampleStr
      }).seq.toArray
    Utils.createParentDirs(encodedDataFile)
    val printWriter: PrintWriter = new PrintWriter(new File(encodedDataFile))
    printWriter.println(lines.mkString("\n"))
    printWriter.close
  }

  /**
   * Reads encoded data.
   */
  def readEncodedData(encodedDataFile: String): Array[LabeledPoint] = {
    Source.fromFile(new File(encodedDataFile)).getLines.toArray.
    map(sample => {
      val values: Array[Double] = sample.split("\t", -1).map(_.toDouble)
      LabeledPoint(values(0), values.drop(1))
    })
  }

  /**
   * Finds cardinalities of categorical features. Assumes a categorical
   * feature of cardinality K takes values in {0,1,...,K-1},
   * and thus finds the maximum index of each categorical feature
   * in the data.
   */
  def findCardinalitiesForFeatures(featureTypes: Array[Int], data: Array[LabeledPoint]): Array[Int] = {
    val numFeatures: Int = featureTypes.length
    val cardinalitiesForFeatures: Array[Int] = new Array(numFeatures)
    data.foreach(sample => {
      val features: Array[Double] = sample.features
      for (j <- 0 to numFeatures - 1) {
        if (featureTypes(j) == 1) {
          cardinalitiesForFeatures(j) = math.max(cardinalitiesForFeatures(j), features(j).toInt)
        }
      }
    })
    for (j <- 0 to numFeatures - 1) {
      if (featureTypes(j) == 1) {
        cardinalitiesForFeatures(j) += 1
      }
    }
    cardinalitiesForFeatures
  }

}
