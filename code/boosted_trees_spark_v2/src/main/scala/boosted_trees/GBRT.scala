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

import java.io.File
import java.io.PrintWriter

import scala.io.Source
import scala.collection.mutable.Stack
import scala.collection.mutable.{Map => MuMap}

/**
 * Functions for training a GBRT model, i.e., a forest
 * of gradient boosted regression trees, printing/saving/reading it and
 * using it for prediction.
 */

object GBRT {
	
	// 1. Functions for training a forest model.
	
	// 1.1. Function for scaling the response of a tree by a shrinkage factor.
	
	def shrinkTree(rootNode : Node, shrinkage : Double) {
		var nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			var node : Node = nodesStack.pop
			node.response = shrinkage * node.response
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
	}
	
	
	// 1.2. Function for training a forest of trees using gradient boosting.
	
	def trainForest(samples : List[Array[Double]], featureTypes : Array[Int],
			numValuesForFeatures : Array[Int], featureWeights : Array[Double],
			numTrees : Int = 5, shrinkage : Double = 0.8,
			maxDepth : Int = 4, minGainFraction : Double = 0.01,
			minLocalGainFraction : Double = 1,
			useSampleWeights : Int = 0) : Array[Node] = {
		val residualSamples : List[Array[Double]] = samples.par.map(_.clone).toList
				// Create a copy of samples which will be modified over iterations.
		val rootNodes : Array[Node] = new Array(numTrees)
		for (m <- 0 to numTrees - 1) {
			println("    Training Tree # " + m + ".")
//			// Old.
//			val residualSamples : List[Array[Double]] = samples.par.map(sample => {
//				val residual : Double = sample(0) - 
//						GBRT.predict(sample, rootNodes.dropRight(numTrees - m))
//				val residualSample : Array[Double] = sample.clone
//					// Without clone, original sample is modified.
//					// May remove for performance.
//				residualSample(0) = residual
//				residualSample
//			}).toList
			// New.
			if (m > 0) {
				residualSamples.par.map(sample => {
					sample(0) = sample(0) - 
							RegressionTree.predict(sample, rootNodes(m-1))
				})
			}
			val rootNode = RegressionTree.trainTree(residualSamples, featureTypes,
					numValuesForFeatures, featureWeights,
					maxDepth, minGainFraction, minLocalGainFraction,
					useSampleWeights)
			GBRT.shrinkTree(rootNode, shrinkage)
			rootNodes(m) = rootNode
		}
		rootNodes
	}
	
	
	// 2. Functions for predicting output for a test sample using a forest model.
	
	// 2.1. Standard regression forest prediction.
	def predict(testSample : Array[Double], rootNodes : Array[Node]) : Double = {
		var output : Double = 0.0
		for (m <- 0 to rootNodes.length - 1) {
			output += RegressionTree.predict(testSample, rootNodes(m))
		}
		output
	}
	
	// 2.2. Interpret the regression forest prediction as a score and classify
	//      as 0 or 1 based on a threshold.
	//      If original labels were 0-1, regression tree prediction can be
	//      interpreted as probability of 1 (Bernoulli parameter)
	//      and binary prediction can be used for classification.
	def binaryPredict(testSample : Array[Double], rootNodes : Array[Node],
			threshold : Double = 0.5) : Int  = {
		val p : Double = predict(testSample, rootNodes) 
		var b : Int = 0
		if (p > threshold) {
			b = 1
		}
		return b
	}
	
	// 3. Function for finding feature gains.
	
	def evaluateFeatureGains(rootNodes : Array[Node], numFeatures : Int) : Array[Double] = {
		val featureGains : Array[Double] = new Array(numFeatures + 1)
		for (m <- 0 to rootNodes.length - 1) {
			val treeFeatureGains : Array[Double] =
				RegressionTree.evaluateFeatureGains(rootNodes(m), numFeatures)
			for (j <- 1 to numFeatures) {
				featureGains(j) += treeFeatureGains(j)
			}
		}
		for (j <- 1 to numFeatures) {
			featureGains(j) /= rootNodes.length
		}
		// val maxGain : Double = featureGains.max
		// featureGains.map(_ * 100 / maxGain)
		featureGains
	}
	
	// 3.1. Evaluate feature subset gains.
	
	def evaluateFeatureSubsetGains(rootNodes : Array[Node]) : List[(Set[Int], Double)] = {
		val subsetGains : MuMap[Set[Int], Double] = MuMap()
		for (m <- 0 to rootNodes.length - 1) {
			val treeSubsetGains :  Map[Set[Int], Double] =
				RegressionTree.evaluateFeatureSubsetGains(rootNodes(m)).toMap
			for (subset <- treeSubsetGains.keySet) {
				if (subsetGains.contains(subset)) {
					subsetGains(subset) = subsetGains(subset) +
						treeSubsetGains(subset)	
				} else {
					subsetGains(subset) = treeSubsetGains(subset)
				}
			}
		}
		// val maxGain : Double =  subsetGains.toList.map(_._2).max
		// subsetGains.toList.sort(_._2 > _._2).map(x => (x._1, x._2 * 100 / maxGain))
		subsetGains.toList.sort(_._2 > _._2)
	}
	
	// 4. Functions for saving and printing a forest model.
	
	// 4.1. Function to save a forest model in text format for later use.
	
	def saveForest(nodesDir : String, rootNodes : Array[Node]) : Unit = {
		(new File(nodesDir)).mkdirs
		for (m <- 0 to rootNodes.length - 1) {
			RegressionTree.saveTree(nodesDir + "/nodes_" + m + ".txt", rootNodes(m))
		}
		val printWriter : PrintWriter = new PrintWriter(new File(nodesDir + "/num_trees.txt"))
		printWriter.println(rootNodes.length)
		printWriter.close
	}
	
	// 4.2. Functions to print a forest model for easy reading.
	
	def printForest(treesDir : String, rootNodes : Array[Node]) : Unit = {
		(new File(treesDir)).mkdirs
		for (m <- 0 to rootNodes.length - 1) {
			RegressionTree.printTree(treesDir + "/tree_" + m + ".txt", rootNodes(m))
		}
	}
	
	def printForest(treesDir : String, rootNodes : Array[Node], features : Array[String],
			indexes : Array[Map[String, Int]]) : Unit = {
		(new File(treesDir)).mkdirs
		for (m <- 0 to rootNodes.length - 1) {
			RegressionTree.printTree(treesDir + "/tree_" + m + "_details.txt", rootNodes(m), features, indexes)
		}
	}
	
	// 4.3. Function to print Grahpviz DOT files.
	
	def printForestDot(treesDir : String, rootNodes : Array[Node], features : Array[String]) : Unit = {
		(new File(treesDir)).mkdirs
		for (m <- 0 to rootNodes.length - 1) {
			RegressionTree.printTreeDot(treesDir + "/tree_" + m + ".dot", rootNodes(m), features)
		}
	}
	
	
	// 5. Function for reading a forest model.
	
	def readForest(nodesDir : String) : Array[Node] = {
		val numTreesText : Array[String] = Source.fromFile(
				new File(nodesDir + "/num_trees.txt")).getLines.toArray
		val numTrees : Int = numTreesText(0).toInt
		val rootNodes : Array[Node] = new Array(numTrees)
		for (m <- 0 to numTrees - 1) {
			rootNodes(m) = RegressionTree.readTree(nodesDir + "/nodes_" + m + ".txt")
		}
		rootNodes
	}

}