package org.apache.spark.mllib.regression

import scala.collection.mutable.Queue
import scala.collection.mutable.MutableList

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.loss.SquareLossStats
import org.apache.spark.mllib.loss.SquareLoss
import org.apache.spark.mllib.util.DataEncoding

/**
 * Regression tree algorithm that trains a
 * decision tree model based on square loss function.
 * Derived from [[org.apache.spark.mllib.regression.DecisionTreeAlgorithm]]
 * using [[org.apache.spark.mllib.loss.SquareLossStats]]
 * and [[org.apache.spark.mllib.loss.SquareLoss]].
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
 * @param useGlobalQuantiles Use the same quantiles for continuous features
 *        across various splits. Speeds up the training by avoiding quantile
 *        calculations for specific data subsets corresponding to nodes.
 *        Results in minor differences in model and prediction performance.
 * @param useCache Cache the training dataset in memory.
 * @param maxNumQuantiles Maximum number of quantiles to use for any continous feature.
 * @param maxNumQuantileSamples Maximum number of samples used for quantile
 *        calculations, if the dataset is larger than this number.
 */
class RegressionTreeAlgorithm(
      featureTypes: Array[Int] = new Array(2),
      maxDepth: Int = 5,
      minGainFraction: Double = 0.01,
      useGlobalQuantiles: Int = 1,
      useCache: Int = 1,
      maxNumQuantiles: Int = 1000,
      maxNumQuantileSamples: Int = 10000
    ) extends DecisionTreeAlgorithm[SquareLossStats](
      new SquareLoss,
      featureTypes,
      maxDepth,
      minGainFraction,
      useGlobalQuantiles,
      useCache,
      maxNumQuantiles,
      maxNumQuantileSamples
    )

/**
 * Top-level program for training a regression tree model using a given dataset
 * and model parameters.
 */
object RegressionTree {

  def main(args: Array[String]) {

    if (args.length < 6) {
      println("Usage: RegressionTree <master> <input_data_file> <input_header_file>" +
          " <model_output_dir> <max_depth> <min_gain_fraction> [<use_global_quantiles>]")
      System.exit(1)
    }

    val sc = new SparkContext(args(0), "RegressionTree")

    val data = MLUtils.loadLabeledData(sc, args(1))
    // val data = sc.textFile(args(1)).map(_.split("\t").map(_.toDouble)).
    //       map(x => LabeledPoint(x(0), x.drop(1)))

    var features : Array[String] = null
    if (!"".equals(args(2))) {
      features = sc.textFile(args(2)).collect.drop(1)
    } else {
      features = Range(0, data.first.features.length).map("F_" + _.toString).toArray
    }
    val numFeatures : Int = features.length
    val featureTypes : Array[Int] = features.map(feature => {if (feature.endsWith("$")) 1 else 0})
        // 0 -> continuous, 1 -> discrete

    var useGlobalQuantiles = 1
    if (args.length == 7) {
      useGlobalQuantiles = args(6).toInt
    }
    val algorithm = new RegressionTreeAlgorithm(featureTypes,
      maxDepth = args(4).toInt, minGainFraction = args(5).toDouble,
      useGlobalQuantiles = useGlobalQuantiles)

    val model = algorithm.train(data)

    sc.parallelize(model.explain, 1).saveAsTextFile(args(3) + "/tree.txt")
    sc.parallelize(model.save, 1).saveAsTextFile(args(3) + "/tree.json")

    sc.stop()
  }
}

/**
 * Top-level program for testing a regression tree model using a
 * test dataset with labels.
 */
object RegressionTreeTest {

  def main(args: Array[String]) {

    if (args.length != 4) {
      println("Usage: RegressionTreeTest <master> <input_data_file> <model_file> <error_dir>")
      System.exit(1)
    }

    val sc = new SparkContext(args(0), "RegressionTreeTest")

    val data = MLUtils.loadLabeledData(sc, args(1))
    // val data = sc.textFile(args(1)).map(_.split("\t").map(_.toDouble)).
    //       map(x => LabeledPoint(x(0), x.drop(1)))

    val model = new DecisionTreeModel
    model.load(sc.textFile(args(2)).collect)

    val predictionsVsExpected = data.map(labeledPoint =>
        (model.predict(labeledPoint.features), labeledPoint.label))

    val lossStats: (Long, Double, Double) = predictionsVsExpected.
        map(pl => (1L, (pl._1 - pl._2) * (pl._1 - pl._2), math.abs(pl._1 - pl._2))).
        reduce((s1, s2) => (s1._1 + s2._1, s1._2 + s2._2, s1._3 + s2._3))
    val rmse: Double = math.sqrt(lossStats._2 / lossStats._1)
    val mae: Double = lossStats._3 / lossStats._1
    val count: Long = lossStats._1

    val lines: MutableList[String] = MutableList()
    lines += "RMSE = " + "%.5g".format(rmse)
    lines += "MAE = " + "%.5g".format(mae)
    lines += "Count = " + count

    println(lines.mkString("\n"))
    sc.parallelize(lines, 1).saveAsTextFile(args(3) + "/error.txt")

    sc.stop()
  }
}

