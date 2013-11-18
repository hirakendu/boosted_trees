/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification

import scala.collection.mutable.Queue
import scala.collection.mutable.MutableList

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.regression.DecisionTreeAlgorithm
import org.apache.spark.mllib.regression.DecisionTreeModel
import org.apache.spark.mllib.loss.EntropyLossStats
import org.apache.spark.mllib.loss.EntropyLoss
import org.apache.spark.mllib.util.DataEncoding

/**
 * Classification tree algorithm that trains a
 * decision tree model based on entropy loss function.
 * The instance labels have to be 0 or 1.
 * The model predictions are scores in <code>[0,1]</code> range,
 * indicating the likelihood that the label is 1.
 * The predictions thus have to be thresholded, say at <code>0.5</code>
 * to get binary label predictions.
 *
 * Derived from [[org.apache.spark.mllib.regression.DecisionTreeAlgorithm]]
 * using [[org.apache.spark.mllib.loss.EntropyLossStats]]
 * and [[org.apache.spark.mllib.loss.EntopyLoss]].
 *
 * @param featureTypes Indicates the types of the features, i.e., whether they
 *        are continuous, <code>0</code> or categorical, <code>1</code>.
 *        Default set to <code>Array(0, 0)</code> (two continuous features),
 *        but mandatory.
 * @param maxDepth Maximum depth of this tree. Used as a condition for
 *        terminating the tree growing procedure.
 * @param minGainFraction Minimum gain of any internal node relative
 *        to the variance/impurity of the root node. Used as a condition for
 *        terminating the tree growing procedure.
 * @param useCache Cache the training dataset in memory.
 * @param maxNumQuantiles Maximum number of quantiles to use for any continous feature.
 * @param maxNumQuantileSamples Maximum number of samples used for quantile
 *        calculations, if the dataset is larger than this number.
 */
class ClassificationTreeAlgorithm(
      featureTypes: Array[Int] = new Array(2),
      maxDepth: Int = 5,
      minGainFraction: Double = 0.01,
      useCache: Int = 1,
      maxNumQuantiles: Int = 1000,
      maxNumQuantileSamples: Int = 10000,
      histogramsMethod: String = "mapreduce"
    ) extends DecisionTreeAlgorithm[EntropyLossStats](
      new EntropyLoss,
      featureTypes,
      maxDepth,
      minGainFraction,
      useCache,
      maxNumQuantiles,
      maxNumQuantileSamples,
      histogramsMethod
    )

/**
 * Top-level program for training a classification tree model
 * using a given dataset and model parameters.
 */
object ClassificationTree {

  def main(args: Array[String]) {

    if (args.length < 6) {
      println("Usage: ClassificationTree <master> <input_data_file> <input_header_file> " +
          "<model_output_dir> <max_depth> <min_gain_fraction> " +
          "[<histograms_method>=mapreduce*|parts]")
      System.exit(1)
    }

    val sc = new SparkContext(args(0), "ClassificationTree")

    val data = MLUtils.loadLabeledData(sc, args(1))
    // val data = sc.textFile(args(1)).map(_.split("\t").map(_.toDouble)).
    //       map(x => LabeledPoint(x(0), x.drop(1)))

    var features : Array[String] = null
    if (!"0".equals(args(2))) {
      features = sc.textFile(args(2)).collect.drop(1)
    } else {
      features = Range(0, data.first.features.length).map("F_" + _.toString).toArray
    }
    val numFeatures : Int = features.length
    val featureTypes : Array[Int] = features.map(feature => {if (feature.endsWith("$")) 1 else 0})
        // 0 -> continuous, 1 -> discrete

    var histogramsMethod = "mapreduce"
    if (args.length >= 7) {
      histogramsMethod = args(6)
    }

    val algorithm = new ClassificationTreeAlgorithm(featureTypes,
      maxDepth = args(4).toInt, minGainFraction = args(5).toDouble,
      histogramsMethod = histogramsMethod)

    val model = algorithm.train(data)

    sc.parallelize(model.explain, 1).saveAsTextFile(args(3) + "/tree.txt")
    sc.parallelize(model.save, 1).saveAsTextFile(args(3) + "/tree.json")

    sc.stop()
  }
}

/**
 * Top-level program for testing a classification tree model using a
 * test dataset with labels.
 */
object ClassificationTreeTest {

  def main(args: Array[String]) {

    if (args.length < 4) {
      println("Usage: ClassificationTreeTest <master> <input_data_file> <model_file> <error_dir> [threshold=0.5]")
      System.exit(1)
    }

    val sc = new SparkContext(args(0), "ClassificationTreeTest")

    val data = MLUtils.loadLabeledData(sc, args(1))
    // val data = sc.textFile(args(1)).map(_.split("\t").map(_.toDouble)).
    //       map(x => LabeledPoint(x(0), x.drop(1)))

    val model = new DecisionTreeModel
    model.load(sc.textFile(args(2)).collect)

    var threshold = 0.5
    if (args.length == 5) {
      threshold = args(4).toDouble
    }

    val predictionsVsExpected = data.map(labeledPoint =>
        (model.predict(labeledPoint.features), labeledPoint.label))

    val lossStats: (Long, Long, Long, Long) = predictionsVsExpected.
        map(pl => {
          var b: Long = 0
          if (pl._1 > threshold) {
            b = 1
          }
          val s: (Long, Long, Long, Long) = pl._2.toInt match {
            case 0 => (1L, b, 0L, 0)
            case 1 => (0L, 0, 1L, 1 - b)
          }
          s
        }).
        reduce((s1, s2) => (s1._1 + s2._1, s1._2 + s2._2, s1._3 + s2._3, s1._4 + s2._4))
    val fp : Long =  lossStats._2
    val fn : Long =  lossStats._4
    val tp : Long =  lossStats._3 - fn
    val tn : Long =  lossStats._1 - fp
    val tpr: Double = tp.toDouble / (tp + fn)
    val fpr: Double = fp.toDouble / (tn + fp)
    val precision: Double = tp.toDouble / (tp + fp)
    val fScore: Double = 2 * tp.toDouble / (2 * tp + fn + fp)
    val accuracy: Double = (tn + tp).toDouble / (tn + tp + fn + fp)

    val lines: MutableList[String] = MutableList()
    lines += "TPR = Recall = " + tp + "/" + (tp + fn) + " = " +
        "%.5g".format(tpr)
    lines += "FPR = " + fp + "/" + (tn + fp) + " = " +
        "%.5g".format(fpr)
    lines += "Precision = " + tp + "/" + (tp + fp) + " = " +
        "%.5g".format(precision)
    lines += "F-score = " +  "%.5g".format(fScore)
    lines += "Accuracy = " + (tn + tp) + "/" + (tn + tp + fn + fp) + " = " +
          "%.5g".format(accuracy)

    println(lines.mkString("\n"))
    sc.parallelize(lines, 1).saveAsTextFile(args(3) + "/error.txt")

    sc.stop()
  }
}

