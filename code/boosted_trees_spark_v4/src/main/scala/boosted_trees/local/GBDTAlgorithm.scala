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

package boosted_trees.local

import scala.collection.mutable.Stack

import boosted_trees.loss.LossStats
import boosted_trees.loss.Loss

import boosted_trees.LabeledPoint

import boosted_trees.GBDTAlgorithmParameters
import boosted_trees.DecisionTreeAlgorithmParameters
import boosted_trees.Node
import boosted_trees.DecisionTreeModel
import boosted_trees.GBDTModel


class GBDTAlgorithm[S <: LossStats[S]:Manifest](
      val loss: Loss[S],
      val params: GBDTAlgorithmParameters
    ) extends Serializable {

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

  def train(data: Array[LabeledPoint]): GBDTModel = {

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
      println("      Finding quantiles for continuous features.")
      val initialTime: Long = System.currentTimeMillis

      params.quantilesForFeatures = DataEncoding.findQuantilesForFeatures(data,
          params.featureTypes, params.maxNumQuantiles)

      val finalTime: Long = System.currentTimeMillis
      println("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    // 3. Find cardinalities of categorical features.

    if ("aggregate".equals(params.histogramsMethod) && params.cardinalitiesForFeatures == null) {
      println("      Finding cardinalities for categorical features.")
      val initialTime: Long = System.currentTimeMillis

      params.cardinalitiesForFeatures = DataEncoding.findCardinalitiesForFeatures(params.featureTypes, data)

      val finalTime: Long = System.currentTimeMillis
      println("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    // 4. Gradient boosting iterations.

    var residualData: Array[LabeledPoint] = data
    val trees: Array[DecisionTreeModel] = new Array(params.numTrees)
    for (m <- 0 to params.numTrees - 1) {
      println("    Training Tree # " + m + ".")
      val initialTime: Long = System.currentTimeMillis

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
          useCache = 0,  // Not applicable.
          params.cardinalitiesForFeatures,
          params.useGlobalQuantiles,
          params.quantilesForFeatures,
          params.maxNumQuantiles,
          params.maxNumQuantileSamples,
          params.histogramsMethod,
          params.batchSize,
          params.numReducersPerNode,  // Not applicable.
          params.maxNumReducers  // Not applicable.
        )

      if (params.fastTree == 1) {
        val algorithm: FastDecisionTreeAlgorithm[S] =
          new FastDecisionTreeAlgorithm[S](loss, dtParams)
        trees(m) = algorithm.train(residualData)
      } else {  // fastTree == 0
        val algorithm: SimpleDecisionTreeAlgorithm[S] =
          new SimpleDecisionTreeAlgorithm[S](loss, dtParams)
        trees(m) = algorithm.train(residualData)
      }

      shrinkTree(trees(m))

      residualData = residualData.map(sample => {
          LabeledPoint(sample.label - trees(m).predict(sample.features),
              sample.features)
        })

      val finalTime: Long = System.currentTimeMillis
      println("        Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    new GBDTModel(trees)
  }

}
