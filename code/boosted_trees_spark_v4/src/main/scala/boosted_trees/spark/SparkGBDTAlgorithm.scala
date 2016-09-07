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

package boosted_trees.spark

import scala.collection.mutable.Stack

import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.storage.StorageLevel

import boosted_trees.loss.LossStats
import boosted_trees.loss.Loss

import boosted_trees.LabeledPoint

import boosted_trees.GBDTAlgorithmParameters
import boosted_trees.DecisionTreeAlgorithmParameters
import boosted_trees.Node
import boosted_trees.DecisionTreeModel
import boosted_trees.GBDTModel


class SparkGBDTAlgorithm[S <: LossStats[S]:Manifest](
      val loss: Loss[S],
      val params: GBDTAlgorithmParameters
    ) extends Logging with Serializable {

  /**
   * Helper function for scaling the response of a tree by a shrinkage factor.
   */
  def shrinkTree(tree: DecisionTreeModel) {
    var nodesStack: Stack[Node] = Stack()
    nodesStack.push(tree.rootNode)
    while (!nodesStack.isEmpty) {
      var node: Node = nodesStack.pop
      node.response = params.shrinkage * node.response
      if (!node.isLeaf) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
  }

  def train(data: RDD[LabeledPoint]): GBDTModel = {

    // 1. Ensure feature types and feature weights are set,
    //    and adjust for sample weight field.

    val numFeatures: Int = params.featureTypes.length
    if (params.featureTypes == null) {
      params.featureTypes = Array.fill[Int](numFeatures)(0)
    }
    if (params.featureWeights == null) {
      params.featureWeights = Array.fill[Double](numFeatures)(1.0)
    }
    if (params.useSampleWeights == 1) {
      params.featureWeights(numFeatures - 1) = 0.0
      params.featureTypes(numFeatures - 1) = 0
    }

    // 2. Find quantiles for continuous features.

    if ((params.fastTree == 1 || params.useGlobalQuantiles == 1) && params.quantilesForFeatures == null) {
      logInfo("      Finding quantiles for continuous features.")
      val initialTime: Long = System.currentTimeMillis

      params.quantilesForFeatures = SparkDataEncoding.findQuantilesForFeatures(data,
          params.featureTypes, params.maxNumQuantiles, params.maxNumQuantileSamples)

      val finalTime: Long = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    // 3. Find cardinalities of categorical features.

    if ("aggregate".equals(params.histogramsMethod) && params.cardinalitiesForFeatures == null) {
      logInfo("      Finding cardinalities for categorical features.")
      val initialTime: Long = System.currentTimeMillis

      params.cardinalitiesForFeatures = SparkDataEncoding.findCardinalitiesForFeatures(params.featureTypes, data)

      val finalTime: Long = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    // 4. Gradient boosting iterations.

    var residualData: RDD[LabeledPoint] = data
    var oldPersistedResidualData: RDD[LabeledPoint] = null
    val trees: Array[DecisionTreeModel] = new Array(params.numTrees)
    for (m <- 0 to params.numTrees - 1) {
      logInfo("    Training Tree # " + m + ".")
      val initialTime: Long = System.currentTimeMillis

      if (params.useCache == 1 && m % params.persistInterval == 0) {
        residualData.persist(StorageLevel.MEMORY_AND_DISK)
        // residualData.persist(StorageLevel.MEMORY_AND_DISK_2)
        // residualData.persist(StorageLevel.MEMORY_AND_DISK_SER)
        // residualData.persist
        // residualData.foreach(sample => {})  // Materialize before uncaching parent.
        if (m > 0) {
          oldPersistedResidualData.unpersist(true)
        }
        oldPersistedResidualData = residualData
      }

      val dtParams: DecisionTreeAlgorithmParameters =
        DecisionTreeAlgorithmParameters(
          params.featureTypes,
          params.maxDepth,
          params.minGainFraction,
          params.minLocalGainFraction,
          params.minWeight,
          params.minCount,
          params.featureWeights,
          params.useSampleWeights,
          useCache = 0,
          params.cardinalitiesForFeatures,
          params.useGlobalQuantiles,
          params.quantilesForFeatures,
          params.maxNumQuantiles,
          params.maxNumQuantileSamples,
          params.histogramsMethod,
          params.batchSize,
          params.numReducersPerNode,
          params.maxNumReducers
        )

      if (params.fastTree == 1) {
        val algorithm: SparkFastDecisionTreeAlgorithm[S] =
          new SparkFastDecisionTreeAlgorithm[S](loss, dtParams)
        trees(m) = algorithm.train(residualData)
      } else {  // fastTree == 0
        val algorithm: SparkSimpleDecisionTreeAlgorithm[S] =
          new SparkSimpleDecisionTreeAlgorithm[S](loss, dtParams)
        trees(m) = algorithm.train(residualData)
      }

      shrinkTree(trees(m))

      val oldResidualData = residualData
      residualData = oldResidualData.map(sample => {
          LabeledPoint(sample.label - trees(m).predict(sample.features),
              sample.features)
        })

      val finalTime: Long = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    new GBDTModel(trees)
  }

}
