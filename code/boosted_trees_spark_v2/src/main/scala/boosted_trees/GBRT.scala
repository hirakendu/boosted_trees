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
			numTrees : Int = 5, shrinkage : Double = 0.8,
			maxDepth : Int = 4, minGainFraction : Double = 0.01) : Array[Node] = {
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
					maxDepth, minGainFraction)
			GBRT.shrinkTree(rootNode, shrinkage)
			rootNodes(m) = rootNode
		}
		rootNodes
	}
	
	
	// 2. Function for predicting output for a test sample using a forest model.
	
	def predict(testSample : Array[Double], rootNodes : Array[Node]) : Double = {
		var output : Double = 0.0
		for (m <- 0 to rootNodes.length - 1) {
			output += RegressionTree.predict(testSample, rootNodes(m))
		}
		output
	}
	
	
	// 3. Functions for saving and printing a forest model.
	
	// 3.1. Function to save a forest model in text format for later use.
	
	def saveForest(nodesDir : String, rootNodes : Array[Node]) : Unit = {
		(new File(nodesDir)).mkdirs
		for (m <- 0 to rootNodes.length - 1) {
			RegressionTree.saveTree(nodesDir + "/nodes_" + m + ".txt", rootNodes(m))
		}
		val printWriter : PrintWriter = new PrintWriter(new File(nodesDir + "/num_trees.txt"))
		printWriter.println(rootNodes.length)
		printWriter.close
	}
	
	// 3.2. Functions to print a forest model for easy reading.
	
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
	
	// 3.3. Function to print Grahpviz DOT files.
	
	def printForestDot(treesDir : String, rootNodes : Array[Node], features : Array[String]) : Unit = {
		(new File(treesDir)).mkdirs
		for (m <- 0 to rootNodes.length - 1) {
			RegressionTree.printTreeDot(treesDir + "/tree_" + m + "_details.dot", rootNodes(m), features)
		}
	}
	
	
	// 4. Function for reading a forest model.
	
	def readForest(nodesDir : String) : Array[Node] = {
		val numTreesText : Array[String] = Source.fromFile(
				new File(nodesDir + "/num_trees.txt")).getLines.toArray
		val numTrees : Int = numTreesText(0).toInt
		val rootNodes : Array[Node] = new Array(numTrees)
		for (m <- 0 to numTrees - 1) {
			rootNodes(m) = RegressionTree.readTree(nodesDir + "nodes_" + m + ".txt")
		}
		rootNodes
	}

}