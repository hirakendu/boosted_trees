package org.apache.spark.mllib.util

import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.regression.Node
import org.apache.spark.mllib.regression.DecisionTreeModel

object DecisionTreeDataGenerator {

  def generateDecisionTreeInput(
      featureTypes: Array[Int],
      numBinsForFeatures: Array[Int],
      rootNode: Node,
      nPoints: Int,
      seed: Int,
      eps: Double = 0.1): Seq[LabeledPoint] = {

    val rnd = new Random(seed)
    val model: DecisionTreeModel = new DecisionTreeModel(rootNode)
    val data: Array[LabeledPoint] = new Array(nPoints)
    for (i <- 0 to nPoints - 1) {
      val features: Array[Double] = new Array(featureTypes.length)
      for (j <- 0 to featureTypes.length - 1) {
        features(j) = rnd.nextDouble
        if (featureTypes(j) == 1) {
          features(j) = (features(j) * numBinsForFeatures(j)).toInt
        }
      }
      var label = model.predict(features)
      if (rnd.nextDouble < eps) {
        label = 1 - label
      }
      data(i) = LabeledPoint(label, features)
    }
    data
  }
  
  def generateDecisionTreeRDD(
      sc: SparkContext,
      nexamples: Int,
      eps: Double,
      nparts: Int = 2) : RDD[LabeledPoint] = {
    // TODO: Generate random tree.
    // Currently using  Y = 1(X0 >= 0.5 && X1 == 1),
    // where X0 in [0,1] and X1 in {0,1,2}.
    val featureTypes: Array[Int] = Array(0, 1)
    val nodes: Array[Node] = new Array(7)
    for (i <- 0 to nodes.length - 1) {
      nodes(i) = new Node
    }
    nodes(0).featureId = 1
    nodes(0).featureType = 1
    nodes(0).leftBranchValues = Set(0,2)
    nodes(0).leftChild = nodes(1)
    nodes(0).rightChild = nodes(2)
    nodes(1).response = 0
    nodes(2).featureId = 0
    nodes(2).featureType = 0
    nodes(2).threshold = 0.5
    nodes(2).leftChild = nodes(5)
    nodes(2).rightChild = nodes(6)
    nodes(5).response  = 0
    nodes(6).response = 1
    val rootNode: Node = nodes(0)
    val numBinsForFeatures: Array[Int] = Array(0, 3)

    val data: RDD[LabeledPoint] = sc.parallelize(0 until nparts, nparts).flatMap { p =>
      val seed = 42 + p
      val examplesInPartition = nexamples / nparts
      generateDecisionTreeInput(featureTypes, numBinsForFeatures,
          rootNode, examplesInPartition, seed, eps)
    }
    data
  }

  def main(args: Array[String]) {
    if (args.length < 2) {
      println("Usage: DecisionTreeDataGenerator " +
        "<master> <output_dir> [num_examples] [num_partitions]")
      System.exit(1)
    }

    val sparkMaster: String = args(0)
    val outputPath: String = args(1)
    val nexamples: Int = if (args.length > 2) args(2).toInt else 1000
    val parts: Int = if (args.length > 4) args(4).toInt else 2
    val eps = 0.1

    val sc = new SparkContext(sparkMaster, "DecisionTreeDataGenerator")
    val data = generateDecisionTreeRDD(sc, nexamples, eps, nparts = parts)

    MLUtils.saveLabeledData(data, outputPath)
    sc.stop()
  }
}