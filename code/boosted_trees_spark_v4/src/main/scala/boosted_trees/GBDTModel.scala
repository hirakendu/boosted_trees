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

package boosted_trees

import scala.collection.mutable.MutableList
import scala.collection.mutable.{Map => MuMap}

import com.google.gson.JsonParser
import com.google.gson.JsonObject
import com.google.gson.JsonArray


class GBDTModel(
    var trees: Array[DecisionTreeModel],
    var features: Array[String] = null,
    var indexes: Array[Map[String, Int]] = null
    ) extends Serializable {

  def this(numTrees: Int) = this(new Array(numTrees), null, null)

  def this() = this(1)

  def predict(features: Array[Double]): Double = {
    var output: Double = 0.0
    for (m <- 0 to trees.length - 1) {
      output += trees(m).predict(features)
    }
    output
  }

  def predict(features: Array[String]): Double = {
    var output: Double = 0.0
    for (m <- 0 to trees.length - 1) {
      output += trees(m).predict(features)
    }
    output
  }

  def binaryPredict(features: Array[Double], threshold: Double): Int = {
    if (predict(features) < threshold) 0 else 1
  }

  def binaryPredict(features: Array[String], threshold: Double): Int = {
    if (predict(features) < threshold) 0 else 1
  }

  def loadNamesForIds(features: Array[String], indexes: Array[Map[String, Int]]): Unit = {
    this.features = features.clone
    this.indexes = indexes.clone
    for (m <- 0 to trees.length - 1) {
      trees(m).loadNamesForIds(features, indexes)
    }
  }

  def evaluateFeatureGains(): Array[Double] = {
    val featureGains: Array[Double] = new Array(features.length)
    for (m <- 0 to trees.length - 1) {
      val treeFeatureGains: Array[Double] = trees(m).evaluateFeatureGains()
      for (j <- 0 to features.length - 1) {
        featureGains(j) += treeFeatureGains(j)
      }
    }
    featureGains
  }

  def evaluateFeatureImportances(): Array[(String, Double)] = {
    val featureGains: Array[Double] = evaluateFeatureGains()
    val maxGain = featureGains.max
    featureGains.zipWithIndex.map(x => (features(x._2), x._1 * 100 / maxGain)).
        sortWith(_._2 > _._2)
  }

  def evaluateFeatureSubsetGains(): Array[(Set[String], Double)] = {
    val subsetGains: MuMap[Set[String], Double] = MuMap()
    for (m <- 0 to trees.length - 1) {
      val treeSubsetGains:  Map[Set[String], Double] =
        trees(m).evaluateFeatureSubsetGains().toMap
      for (subset <- treeSubsetGains.keySet) {
        if (subsetGains.contains(subset)) {
          subsetGains(subset) = subsetGains(subset) + treeSubsetGains(subset)
        } else {
          subsetGains(subset) = treeSubsetGains(subset)
        }
      }
    }
    subsetGains.toArray.sortWith(_._2 > _._2)
  }

  def evaluateFeatureSubsetImportances(): Array[(Set[String], Double)] = {
    val featureSubsetGains: Array[(Set[String], Double)] = evaluateFeatureSubsetGains()
    val maxGain: Double = featureSubsetGains.map(_._2).max
    featureSubsetGains.map(x => (x._1, x._2 * 100 / maxGain))
  }

  def explain(indent: String = ""): Array[String] = {
    val lines: MutableList[String] = MutableList()
    for (m <- 0 to trees.length - 1) {
      lines += "m = " + m
      lines ++= trees(m).explain(indent + "  ")
    }
    lines.toArray
  }

  def save(indent: String = ""): Array[String] = {
    val lines: MutableList[String] = MutableList()
    lines += indent + "{"
    lines += indent + "  \"numTrees\": " + trees.length + ","
    lines += indent + "  \"trees\": " + "["
    for (m <- 0 to trees.length - 1) {
      lines ++= trees(m).save(indent + "    ")
      if (m != trees.length - 1) {
        lines(lines.size - 1) += ","  // Add commas between trees.
      }
    }
    lines += indent + "  ]"
    lines += indent + "}"
    lines.toArray
  }

  def saveExtended(indent: String = ""): Array[String] = {
    val lines: MutableList[String] = MutableList()
    lines += indent + "{"
    lines += indent + "  \"numTrees\": " + trees.length + ","
    lines += indent + "  \"trees\": " + "["
    for (m <- 0 to trees.length - 1) {
      lines ++= trees(m).saveExtended(indent + "    ")
      if (m != trees.length - 1) {
        lines(lines.size - 1) += ","  // Add commas between trees.
      }
    }
    lines += indent + "  ]"
    lines += indent + "}"
    lines.toArray
  }

  def saveSimple(): Array[Array[String]] = {
    val linesForTrees: Array[Array[String]] = new Array(trees.length)
    for (m <- 0 to trees.length - 1) {
      linesForTrees(m) = trees(m).saveSimple()
    }
    linesForTrees
  }

  def printDot(): Array[Array[String]] = {
    val linesForTrees: Array[Array[String]] = new Array(trees.length)
    for (m <- 0 to trees.length - 1) {
      linesForTrees(m) = trees(m).printDot()
    }
    linesForTrees
  }

  def load(lines: Array[String]): Unit = {
    val parser: JsonParser = new JsonParser
    val treesJson: JsonArray = parser.parse(lines.mkString).
        getAsJsonObject.get("trees").getAsJsonArray
    load(treesJson)
  }

  def load(treesJson: JsonArray): Unit = {
    trees = new Array(treesJson.size)
    for (m <- 0 to trees.length - 1) {
      trees(m) = new DecisionTreeModel
      trees(m).load(treesJson.get(m).
          getAsJsonObject.get("nodes").getAsJsonArray)
    }
  }

  def loadExt(lines: Array[String]): Unit = {
    val parser: JsonParser = new JsonParser
    val treesJson: JsonArray = parser.parse(lines.mkString).
        getAsJsonObject.get("trees").getAsJsonArray
    loadExt(treesJson)
  }

  def loadExt(treesJson: JsonArray): Unit = {
    trees = new Array(treesJson.size)
    for (m <- 0 to trees.length - 1) {
      trees(m) = new DecisionTreeModel
      trees(m).loadExt(treesJson.get(m).
          getAsJsonObject.get("nodes").getAsJsonArray)
    }
  }

  def loadSimple(linesForTrees: Array[Array[String]]) {
    trees = new Array(linesForTrees.length)
    for (m <- 0 to trees.length - 1) {
      trees(m) = new DecisionTreeModel
      trees(m).loadSimple(linesForTrees(m))
    }
  }

}
