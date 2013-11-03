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
import scala.collection.mutable.{Set => MuSet}
import scala.collection.mutable.{Map => MuMap}
import scala.collection.mutable.MutableList
import scala.collection.parallel.immutable.ParSeq
import scala.collection.mutable.Stack
import scala.util.control.Breaks._

/**
 * Functions for training a regression tree model,
 * printing/saving/reading it and using it for prediction.
 */

object RegressionTree {
	
	// 1. Functions for training a tree model.
	
	// 1.1. Function for training a single node, i.e., finding best split
	//      for a given dataset.
	
	def trainNode(node : Node, samples : Array[Array[Double]],
			featureTypes : Array[Int], numValuesForFeatures : Array[Int],
			featureWeights : Array[Double],
			maxDepth : Int = 4, minGain : Double = 1e-6,
			minLocalGainFraction : Double = 1, useSampleWeights: Int = 0) :
			(Array[Array[Double]], Array[Array[Double]]) = {
		
		// 0. Parameters.
		val maxNumQuantileValues : Int = 1000  // Q-1. At least 1, i.e., Q >= 2.
		val maxNumQuantileSamples : Int = 10000
		
		// 1. Some node statistics, prior to training.
		val numFeatures : Int = featureTypes.length
		if (useSampleWeights == 0) {
			val stats : (Long, Double, Double) = samples.
				map(sample => (1L, sample(0), sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
			node.numSamples = stats._1
			node.weight = stats._1
			node.response = stats._2 / stats._1
			node.error = stats._3 - stats._2 * stats._2 / stats._1
		} else {
			val stats : (Long, Double, Double, Double) = samples.
				map(sample => (1L, sample(numFeatures - 1),
						sample(numFeatures - 1) * sample(0),
						sample(numFeatures - 1) * sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3,
						stats1._4 + stats2._4))
			node.numSamples = stats._1
			node.weight = stats._2
			node.response = stats._3 / stats._2
			node.error = stats._4 - stats._3 * stats._3 / stats._2
		}


				
		// 2. Don't split if not enough samples or depth too high
		//    or no variable to split.
		//    or other due to other termination criteria.
		//    Additional criteria are used to stop the split further along.
		if (node.numSamples < 2) {
			return (null, null)
		}
		if (node.id >= math.pow(2, maxDepth)) {
			return (null, null)
		}
		
		// 3. Find best split for each feature.
		val numSamples : Long = node.numSamples
		val errorsForFeatures : Array[Double] = new Array(numFeatures)
		val thresholdsForFeatures : Array[Double] = new Array(numFeatures)
		val leftValuesForFeatures : Array[Set[Int]]  = new Array(numFeatures)
		val rightValuesForFeatures : Array[Set[Int]]  = new Array(numFeatures)
		
		// 3.1. Find the candidate thresholds for continuous features.
		val candidateThresholdsForFeatures : Array[Array[Double]] = new Array(numFeatures)
		val numQuantileSamples : Int = math.min(maxNumQuantileSamples.toLong, node.numSamples).toInt
		val numQuantileValues : Int = math.min(maxNumQuantileValues, numQuantileSamples - 1)
		val quantileSampleIds : Set[Long] = Utils.sampleWithoutReplacement(numSamples, numQuantileSamples)
		val quantileSamples : Array[Array[Double]] = samples.zipWithIndex.
			filter(sampleId => quantileSampleIds.contains(sampleId._2)).map(_._1)
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			// Skip i = 0 corresponding to label.
			if (featureTypes(j) == 0) {
				val featureValues : Array[Double] = quantileSamples.map(_(j)).sortWith(_ < _)
				val candidateThresholdsSet : MuSet[Double] = MuSet()
				for (q <- 1 to numQuantileValues) {
					val id : Int = (q * numQuantileSamples / (numQuantileValues + 1.0) - 1).toInt
					if (id >= 0) {
						val threshold : Double = (featureValues(id) + featureValues(id + 1)) / 2.0
						candidateThresholdsSet.add(threshold)
					}
				}
				candidateThresholdsForFeatures(j) =
					candidateThresholdsSet.toArray.sortWith(_ < _)
			}
		// }
		})  // End of j-loop for finding candidate thresholds for continuous features.
		
		// 3.2. Find the stats (histogram, mean response, mean square response) 
		//      for various feauture-value bins. Most data intensive.
//		// Old method: giant reduce.
//		val statsForFeatureBins : Array[((Int, Int), (Long, Double, Double))] =
//			samples.flatMap(sample => {
//				val square : Double = sample(0) * sample(0)
//				val valueBinsForFeatures : Array[Int] = new Array(numFeatures)
//				// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
//				for (j <- 1 to numFeatures - 1) {
//					var bj : Int = 0  // Bin index of the value.
//					if (featureTypes(j) == 0) {
//						// Continuous feature.
//						val numBins : Int =  candidateThresholdsForFeatures(j).length + 1
//						bj = numBins - 1
//						breakable { for (b <- 0 to numBins - 2) {
//							if (sample(j) < candidateThresholdsForFeatures(j)(b)) {
//								bj = b
//								break
//							}
//						} }
//					} else if (featureTypes(j) == 1) {
//						// Discrete feature.
//						bj = sample(j).toInt
//					}
//					valueBinsForFeatures(j) = bj
//				}
//				// })
//				Range(1, numFeatures).zip(valueBinsForFeatures.drop(1)).
//					map(featureBin => MuMap(featureBin -> (1L, sample(0), square)))
//			}).
//			reduce((map1, map2) => {
//				val merged : MuMap[(Int, Int), (Long, Double, Double)] = map1.clone
//				for (key <- map2.keySet) {
//					if (merged.contains(key)) {
//						merged(key) = (merged(key)._1 + map2(key)._1,
//								merged(key)._2 + map2(key)._2,
//								merged(key)._3 + map2(key)._3)
//					} else {
//						merged(key) = map2(key)
//					}
//				}
//				merged
//			}).toArray
		
		// New method. Iterative reduce.
		val statsForFeatureBins : Array[Array[Array[Double]]] = new Array(numFeatures)
		for (j <- 1 to numFeatures - 1) {
			if (featureTypes(j) == 0) {
				statsForFeatureBins(j) = new Array(candidateThresholdsForFeatures(j).length + 1)
			} else {
				statsForFeatureBins(j) = new Array(numValuesForFeatures(j))
			}
			for (b <- 0 to statsForFeatureBins(j).length - 1) {
				statsForFeatureBins(j)(b) = Array(0, 0, 0)
			}
		}
		for (sample <- samples) {
			val square : Double = sample(0) * sample(0)
			// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
			for (j <- 1 to numFeatures - 1) {
				var bj : Int = 0  // Bin index of the value.
				if (featureTypes(j) == 0) {
					// Continuous feature.
					val numBins : Int =  candidateThresholdsForFeatures(j).length + 1
					var b1 : Int = 0
					var b2 : Int = numBins - 1
					while (b1 < b2) {
						val b3 = (b1 + b2) / 2
						if (sample(j) < candidateThresholdsForFeatures(j)(b3)) {
							b2 = b3
						} else {
							b1 = b3 + 1
						}
					}
					bj = b1
//					breakable { for (b <- 0 to numBins - 2) {
//						if (sample(j) < candidateThresholdsForFeatures(j)(b)) {
//							bj = b
//							break
//						}
//					} }
				} else if (featureTypes(j) == 1) {
					// Discrete feature.
					bj = sample(j).toInt
				}
				if (useSampleWeights == 0) {
					statsForFeatureBins(j)(bj)(0) += 1
					statsForFeatureBins(j)(bj)(1) += sample(0)
					statsForFeatureBins(j)(bj)(2) += square
				} else {
					statsForFeatureBins(j)(bj)(0) += sample(numFeatures - 1)
					statsForFeatureBins(j)(bj)(1) += sample(0) * sample(numFeatures - 1)
					statsForFeatureBins(j)(bj)(2) += square * sample(numFeatures - 1)
				}
			}
			// })
		}
			
		// 3.3. Separate the histograms, means, and calculate errors.
		//      Sort bins of discrete features by mean responses.
		//		Note that we can read in parallel from
		//		statsForFeatureBins.
		val statsForBinsForFeatures: Array[Array[(Int, Double, Double, Double)]] = new Array(numFeatures)
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			val statsForBins : Array[(Int, Double, Double, Double)] =
				statsForFeatureBins(j).zipWithIndex.
					map(x => (x._2, x._1(0), x._1(1), x._1(2))).
					filter(_._2 > 0)  // Filter is required to divide by counts.
			if (featureTypes(j) == 0) {
				// For continuous features, order by bin indices,
				// i.e., order of quantiles.
				statsForBinsForFeatures(j) = statsForBins.sortWith(_._1 < _._1)
			} else if (featureTypes(j) == 1) {
				// For categorical features, order by means of bins,
				// i.e., order of means of values.
				statsForBinsForFeatures(j) = statsForBins.
						sortWith((stats1, stats2) => stats1._3/stats1._2 < stats2._3/stats2._2)
			}
		// }
		})
		
		// 3.4. Calculate the best split and error for each feature.
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			
			val statsForBins : Array[(Int, Double, Double, Double)] = statsForBinsForFeatures(j)
			val numBins : Int = statsForBins.length
			
			// Initial split and error.
			var leftSumWeight : Double =  statsForBins(0)._2
			var leftSumResponses : Double = statsForBins(0)._3
			var leftSumSquares : Double = statsForBins(0)._4
			var rightSumWeight : Double = 0
			var rightSumResponses : Double = 0
			var rightSumSquares : Double = 0
			for (b <- 1 to numBins - 1) {
				rightSumWeight += statsForBins(b)._2
				rightSumResponses += statsForBins(b)._3
				rightSumSquares += statsForBins(b)._4
			}
			var bMin : Int = 0
			var minError : Double = 0
			if (leftSumWeight != 0) {
				minError += leftSumSquares - leftSumResponses * leftSumResponses / leftSumWeight
			}
			if (rightSumWeight != 0) {
				minError += rightSumSquares - rightSumResponses * rightSumResponses / rightSumWeight
			}
			
			// Slide threshold from left to right and find best threshold.
			for (b <- 1 to numBins - 2) {
				leftSumWeight +=  statsForBins(b)._2
				leftSumResponses += statsForBins(b)._3
				leftSumSquares += statsForBins(b)._4
				rightSumWeight -= statsForBins(b)._2
				rightSumResponses -= statsForBins(b)._3
				rightSumSquares -= statsForBins(b)._4
				var error : Double = 0
				if (leftSumWeight != 0) {
					error += leftSumSquares - leftSumResponses * leftSumResponses / leftSumWeight
				}
				if (rightSumWeight != 0) {
					error += rightSumSquares - rightSumResponses * rightSumResponses / rightSumWeight
				}
				if (error < minError) {
					minError = error
					bMin = b
				}
			}  // End of b-loop over bins.
			
			errorsForFeatures(j) = minError
			if (featureTypes(j) == 0) {
				// For continuous features, if a bin is empty, it doesn't appear.
				// So threshold id may not equal bin id.
				if (statsForBins(bMin)._1 == candidateThresholdsForFeatures(j).length) {
					// When samples at the right end have same feature value
					// such that the last but one bin is empty and everything goes to the last bin.
					// Note that if x_j >= t, it goes to the bin on right.
					thresholdsForFeatures(j) = candidateThresholdsForFeatures(j)(statsForBins(bMin)._1 - 1)
				} else {
					thresholdsForFeatures(j) = candidateThresholdsForFeatures(j)(statsForBins(bMin)._1)
				}
			} else if  (featureTypes(j) == 1) {
				leftValuesForFeatures(j) = Range(0, bMin + 1).map(statsForBins(_)._1).toSet
				rightValuesForFeatures(j) = Range(bMin + 1, numBins).map(statsForBins(_)._1).toSet
			}
		// }
		})
		
		// 3.5. Find the feature with best split, i.e., maximum weighted gain.
		var jMax : Int = numFeatures - 1
		var maxWeightedGain : Double = -1
		if (useSampleWeights == 0) {
			maxWeightedGain = (node.error - errorsForFeatures(numFeatures - 1)) *
					featureWeights(numFeatures - 1)
		}
		for (j <- 1 to numFeatures - 2) {
			val weightedGain : Double = (node.error - errorsForFeatures(j)) * featureWeights(j)
			if (weightedGain > maxWeightedGain) {
				maxWeightedGain = weightedGain
				jMax = j
			}
		}
		node.featureId = jMax
		node.featureType = featureTypes(jMax)
		if (featureTypes(jMax) == 0) {
			node.threshold = thresholdsForFeatures(jMax)
		} else if (featureTypes(jMax) == 1) {
			node.leftValues = leftValuesForFeatures(jMax)
			node.rightValues = rightValuesForFeatures(jMax)
		}
		node.splitError = errorsForFeatures(jMax)
		node.gain = node.error - node.splitError
		
		// 3.6. Don't split if no gain.
		if (node.gain <= minGain + 1e-7 && node.gain <= minLocalGainFraction * node.error + 1e-7) {
			return (null, null)
		}
		
		// 3.7. Split samples for left and right nodes
		//      and split the child nodes recursively.
		node.leftChild = Some(new Node())
		node.rightChild = Some(new Node())
		node.leftChild.get.parent = Some(node)
		node.rightChild.get.parent = Some(node)
		node.leftChild.get.id = node.id * 2
		node.rightChild.get.id = node.id * 2 + 1
		node.leftChild.get.depth = node.depth + 1 
		node.rightChild.get.depth = node.depth + 1
		var leftSamples : Array[Array[Double]] = null
		var rightSamples : Array[Array[Double]] = null
		if (featureTypes(jMax) == 0) {
			leftSamples = samples.filter(sample => sample(jMax) < node.threshold)
			rightSamples = samples.filter(sample => sample(jMax) >= node.threshold)
		} else if (featureTypes(jMax) == 1) {
			leftSamples = samples.
					filter(sample => node.leftValues.contains(sample(jMax).toInt))
			rightSamples = samples.
					filter(sample => node.rightValues.contains(sample(jMax).toInt))
		}
		
		return (leftSamples, rightSamples)
	}
	
	
	// 1.2. Function for training a tree by recursively training/splitting nodes.
	
	def trainTree(samples: Array[Array[Double]], featureTypes : Array[Int],
			numValuesForFeatures : Array[Int], featureWeights : Array[Double],
			maxDepth : Int = 4, minGainFraction : Double = 0.01,
			minLocalGainFraction : Double = 1,
			useSampleWeights : Int = 0) : Node = {
		val rootNode : Node = new Node
		rootNode.id = 1
		rootNode.depth = 0
		// Find initial error to determine minGain.
		var minGain : Double = 0
		if (useSampleWeights == 0) {
			val stats : (Long, Double, Double) = samples.
				map(sample => (1L, sample(0), sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
			minGain = minGainFraction * (stats._3 - stats._2 * stats._2 / stats._1)
		} else {
			val numFeatures : Int = featureTypes.length
			val stats : (Double, Double, Double) = samples.
				map(sample => (sample(numFeatures - 1),
						sample(numFeatures - 1) * sample(0),
						sample(numFeatures - 1) * sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
			minGain = minGainFraction * (stats._3 - stats._2 * stats._2 / stats._1)		
		}
		// Option 1: Recursive method may cause stack overflow.
		// findBestSplitRecursive(rootNode)
		// Option 2: Iterative by maintaining a queue.
		val nodesStack : Stack[(Node,  Array[Array[Double]])] = Stack()
		nodesStack.push((rootNode, samples))
		while (!nodesStack.isEmpty) {
			val (node, nodeSamples) = nodesStack.pop
			println("      Training Node # " + node.id + ".")
			val (leftSamples, rightSamples) :
				(Array[Array[Double]], Array[Array[Double]]) =
					trainNode(node, nodeSamples, featureTypes,
							numValuesForFeatures, featureWeights,
							maxDepth, minGain, minLocalGainFraction,
							useSampleWeights)
			if (!node.isLeaf) {
				nodesStack.push((node.rightChild.get, rightSamples))
				nodesStack.push((node.leftChild.get, leftSamples))
			}
		}
		rootNode
	}
	
	
	// 2. Functions for predicting output for a test sample using a tree model.
	
	// 2.1. Standard regression tree prediction.
	def predict(testSample : Array[Double], rootNode : Node) : Double = {
		// FIXME: Use tail recursion or stack if possible.
		if (rootNode.isLeaf) {
			rootNode.response
		} else {
			if (rootNode.featureType == 0) {
				// Continuous feature.
				if (testSample(rootNode.featureId) < rootNode.threshold) {
					predict(testSample, rootNode.leftChild.get)
				} else {
					predict(testSample, rootNode.rightChild.get)
				}
			} else {
				// Discrete feature.
				if (rootNode.leftValues.contains(testSample(rootNode.featureId).toInt)) {
					predict(testSample, rootNode.leftChild.get)
				} else {
					predict(testSample, rootNode.rightChild.get)
				}
			}
		}
	}
	
	// 2.2. Interpret the regression tree prediction as a score and classify
	//      as 0 or 1 based on a threshold.
	//      If original labels were 0-1, regression tree prediction can be
	//      interpreted as probability of 1 (Bernoulli parameter)
	//      and binary prediction can be used for classification.
	def binaryPredict(testSample : Array[Double], rootNode : Node,
			threshold : Double = 0.5) : Int  = {
		val p : Double = predict(testSample, rootNode) 
		var b : Int = 0
		if (p > threshold) {
			b = 1
		}
		return b
	}
	
	
	// 3. Function for finding feature gains.
	
	def evaluateFeatureGains(rootNode : Node, numFeatures : Int) : Array[Double] = {
		val featureGains : Array[Double] = new Array(numFeatures + 1)
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			val node : Node = nodesStack.pop
			if (!node.isLeaf) {
				featureGains(node.featureId) += node.gain
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		// val maxGain = featureGains.max
		// featureGains.map(_ * 100 / maxGain)
		featureGains
	}
	
	// 3.1. Evaluate feature subset gains.
	
	def evaluateFeatureSubsetGains(rootNode : Node) : Array[(Set[Int], Double)] = {
		val featureSubsetNodes : MuMap[Set[Int], MuSet[Node]] = MuMap()
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			val node : Node = nodesStack.pop
			val featureSubset : MuSet[Int] = MuSet()
			val nodes : MuSet[Node] = MuSet()
			var parentNode : Option[Node] = node.parent
			while (!parentNode.isEmpty) {
				featureSubset += parentNode.get.featureId
				nodes += parentNode.get
				if (!featureSubsetNodes.contains(featureSubset.toSet)) {
					featureSubsetNodes(featureSubset.toSet) = MuSet()
				}
				featureSubsetNodes(featureSubset.toSet) ++= nodes
				parentNode = parentNode.get.parent
			}
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		val featureSubsetGains : Array[(Set[Int], Double)] =
			featureSubsetNodes.toArray.map(x => (x._1, x._2.map(_.gain).sum)).
			sortWith(_._2 > _._2)
		// val maxGain : Double = featureSubsetGains.map(_._2).max
		// featureSubsetGains.map(x => (x._1, x._2 * 100 / maxGain))
		featureSubsetGains
	}
	
	
	// 4. Functions for saving and printing a tree model.
	
	// 4.1. Functions to save a tree model in text format for later use.
	
	def saveTree(rootNode : Node) : Array[String] = {
		val lines : MutableList[String] = MutableList()
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			// Save head of stack.
			var node : Node = nodesStack.pop
			lines += "id\t" + node.id
			lines += "depth\t" + node.depth
			lines += "response\t%.6f".format(node.response)
			lines += "error\t%.6f".format(node.error)
			lines += "split_error\t%.6f".format(node.splitError)
			lines += "gain\t%.6f".format(node.gain)
			lines += "num_samples\t" + node.numSamples
			lines += "weight\t" + node.weight
			lines += "is_leaf\t" + node.isLeaf
			if (!node.isLeaf) {
				lines += "feature_id\t" + node.featureId
				lines += "feature_type\t" + node.featureType
				if (node.featureType == 0) {
					// Continuous feature.
					lines += "threshold\t%.6f".format(node.threshold)
				} else {
					// Discrete feature.
					lines += "left_values\t" +
							node.leftValues.toArray.sortWith(_ < _).mkString(",")
					lines += "right_values\t" +
							node.rightValues.toArray.sortWith(_ < _).mkString(",")
				}
			}

			// Save children.
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		lines.toArray
	}
	
	def saveTree(nodesFile : String, rootNode : Node) : Unit = {
		val lines : Array[String] = saveTree(rootNode)
		Utils.createParentDirs(nodesFile)
		val printWriter : PrintWriter = new PrintWriter(new File(nodesFile))
		printWriter.println(lines.mkString("\n"))
		printWriter.close
	}
	
	// 4.2. Functions to print a tree model for easy reading.
	
	def printNode(node : Node) : String = {
		var line : String = ""
		for (d <- 1 to node.depth) {
			line += "  "
		}
		line += "i = " + node.id + ", y = " + "%.3f".format(node.response) +
				", n = " + node.numSamples
		if (!node.isLeaf) {
			line += ", x = " + node.featureId + ", t = "
			if (node.featureType == 0) {
				line += "%.3f".format(node.threshold)
			} else {
				line += "{" + node.leftValues.toArray.sorted.mkString(",") + "} : {" +
						node.rightValues.toArray.sorted.mkString(",") + "}"
			}
		}
		line
	}
	
	def printNode(node : Node, features : Array[String],
			reverseIndexes : Array[Map[Int, String]]) : String = {
		var line : String = ""
		for (d <- 1 to node.depth) {
			line += "  "
		}
		line += "i = " + node.id + ", y = " + "%.3f".format(node.response) +
				", n = " + node.numSamples
		if (!node.isLeaf) {
			line += ", x = " + features(node.featureId) +  ", t = "
			if (node.featureType == 0) {
				line += "%.3f".format(node.threshold)
			} else {
				var delimiter = ",\n"
				for (d <- 1 to node.depth + 1) {
					delimiter += "  "
				}
				line += "{" +
						node.leftValues.toArray.map(reverseIndexes(node.featureId)(_)).
							sorted.mkString(delimiter) +
						"} : {" +
						node.rightValues.toArray.map(reverseIndexes(node.featureId)(_)).
							sorted.mkString(delimiter) +
						"}"
			}
		}
		line
	}
	
	def printTree(rootNode : Node) : Array[String] = {
		val lines : MutableList[String] = MutableList()
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			var node : Node = nodesStack.pop
			lines += printNode(node)
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		lines.toArray
	}
	
	def printTree(rootNode : Node, features : Array[String],
			indexes : Array[Map[String, Int]]) : Array[String] = {
		val reverseIndexes : Array[Map[Int, String]] = new Array(features.length)
		for (j <- 1 to features.length - 1) {
			if (features(j).endsWith("$")) {
				reverseIndexes(j) = indexes(j).toArray.map(valueId => (valueId._2, valueId._1)).toMap
			}
		}
		val lines : MutableList[String] = MutableList()
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			var node : Node = nodesStack.pop
			lines += printNode(node, features, reverseIndexes)
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		lines.toArray
	}
	
	def printTree(treeFile : String, rootNode : Node) : Unit = {
		val lines : Array[String] = printTree(rootNode)
		Utils.createParentDirs(treeFile)
		val printWriter : PrintWriter = new PrintWriter(new File(treeFile))
		printWriter.println(lines.mkString("\n"))
		printWriter.close
	}
	
	def printTree(treeFile : String, rootNode : Node, features : Array[String],
			indexes : Array[Map[String, Int]]) : Unit = {
		val lines : Array[String] = printTree(rootNode, features, indexes)
		Utils.createParentDirs(treeFile)
		val printWriter : PrintWriter = new PrintWriter(new File(treeFile))
		printWriter.println(lines.mkString("\n"))
		printWriter.close
	}
	
	// 4.3. Functions to print details of nodes of a tree.
	
	def printNodeDetails(node : Node, features : Array[String],
			reverseIndexes : Array[Map[Int, String]]) : String = {
		var nodeDetails : String = ""
		nodeDetails += "i = " + node.id + "\n"
		nodeDetails += "d = " + node.depth + "\n"
		nodeDetails += "y = " + node.response + "\n"
		if (node.isLeaf) {
			nodeDetails += "x = <leaf>\n"
		} else {
			nodeDetails += "x = " + features(node.featureId) + "\n"
			if (node.featureType == 0) {
				nodeDetails += "t = " + node.threshold + "\n"
			} else if (node.featureType == 1) {
				nodeDetails += "t = {" +
						node.leftValues.toArray.map(reverseIndexes(node.featureId)(_)).
							sorted.mkString(",") +
						"} : {" +
						node.rightValues.toArray.map(reverseIndexes(node.featureId)(_)).
							sorted.mkString(",") +
						"}\n"
			}
		}
		val pathNodes : Stack[Node] = Stack()
		val pathOrientations : Stack[Int] = Stack() 
		val childPositions : Stack[Node] = Stack()
		var parentNode : Node = node
		while (parentNode.parent != None) {
			pathOrientations.push(parentNode.id % 2)
			parentNode = parentNode.parent.get
			pathNodes.push(parentNode)
		}
		nodeDetails += "path =\n"
		while (!pathNodes.isEmpty) {
			parentNode = pathNodes.pop
			val orientation : Int = pathOrientations.pop
			nodeDetails += "  i = " + parentNode.id
			nodeDetails += ", " + features(parentNode.featureId)
			if (parentNode.featureType == 0) {
				if (orientation == 0) {
					nodeDetails += " < "
				} else {
					nodeDetails += " >= "
				}
				nodeDetails += parentNode.threshold + "\n"
			} else if (parentNode.featureType == 1) {
				nodeDetails += " <- {"
				if (orientation == 0) {
					nodeDetails += parentNode.leftValues.toArray.sortWith(_ < _).
							map(reverseIndexes(parentNode.featureId)(_)).mkString(",")
				} else {
					nodeDetails += parentNode.rightValues.toArray.sortWith(_ < _).
							map(reverseIndexes(parentNode.featureId)(_)).mkString(",")
				}
				nodeDetails += "}\n"
			}

		}
		nodeDetails
	}
	
	def printTreeDetails(rootNode : Node, features : Array[String],
			indexes : Array[Map[String, Int]]) : Array[(Int, String)] = {
		val reverseIndexes : Array[Map[Int, String]] = new Array(features.length)
		for (j <- 1 to features.length - 1) {
			if (features(j).endsWith("$")) {
				reverseIndexes(j) = indexes(j).toArray.map(valueId => (valueId._2, valueId._1)).toMap
			}
		}
		val nodesDetails : MutableList[(Int, String)] = MutableList()
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			val node : Node = nodesStack.pop
			nodesDetails += ((node.id, printNodeDetails(node, features, reverseIndexes)))
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		nodesDetails.toArray
	}
	
	def printTreeDetails(treeDetailsDir : String, rootNode : Node,
			features : Array[String], indexes : Array[Map[String, Int]]) : Unit = {
		val nodesDetails : Array[(Int, String)] = printTreeDetails(rootNode, features, indexes)
		(new File(treeDetailsDir)).mkdirs
		for (nodeDetails <- nodesDetails.iterator) {
			val printWriter : PrintWriter = new PrintWriter(new File(treeDetailsDir + "/node_" + nodeDetails._1 + ".txt"))
			printWriter.print(nodeDetails._2)
			printWriter.close
		}
	}
	
	// 4.4. Function to print Grahpviz DOT file.
	
	def printTreeDot(treeFile : String, rootNode : Node, features: Array[String]) : Unit = {
		val printWriter : PrintWriter = new PrintWriter(new File(treeFile))
		printWriter.println("digraph regression_tree {")
		// Print node information.
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			// Print head.
			var node : Node = nodesStack.pop
			var line : String = ""
			for (i <- 0 to node.depth) {
				line += "  "
			}
			line += node.id + " [ shape=record, label=\"{"
			line +=	" i = " + node.id
			if (!node.isLeaf) {
				line +=  " | x = " + features(node.featureId) + ", t ="
				if (features(node.featureId).endsWith("$")) {
					line += node.leftValues.size + ":" + node.rightValues.size
				} else {
					line += node.threshold
				}
			}
			line += " | y = %.2f".format(node.response)
			line += " | n = " + node.numSamples
			line += ", w = %.2f".format(node.weight)
			line += " | e = %.2f".format(math.sqrt(node.error / node.numSamples))
			if (!node.isLeaf) {
				line += ", e' = %.2f".format(math.sqrt(node.splitError / node.numSamples))
			}
			line += " }\" ];"
			printWriter.println(line)
			// Push children.
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		printWriter.println
		// Print connections.
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			// Print head.
			var node : Node = nodesStack.pop
			var line : String = ""
			for (i <- 0 to node.depth) {
				line += "  "
			}
			if (!node.isLeaf) {
				printWriter.println(line + node.id + " -> " + node.leftChild.get.id + "; " +
					node.id + " -> " + node.rightChild.get.id + ";")
			}
			// Push children.
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		printWriter.println("}")
		printWriter.close
		val compileCommand : String = "dot -T pdf " + treeFile + " -o " +
				treeFile.replace(".dot", ".pdf")
		println("  Running system command: "  + compileCommand)
		Runtime.getRuntime().exec(compileCommand).waitFor
	}
	
	
	// 5. Functions for reading a tree model generated by *RegressionTree.
	
	def readTree(linesIter : Iterator[String]) : Node = {
		val rootNode : Node = new Node
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			val node : Node = nodesStack.pop
			node.id = linesIter.next.split("\t")(1).toInt
			node.depth = linesIter.next.split("\t")(1).toInt
			node.response = linesIter.next.split("\t")(1).toDouble
			node.error = linesIter.next.split("\t")(1).toDouble
			node.splitError = linesIter.next.split("\t")(1).toDouble
			node.gain = linesIter.next.split("\t")(1).toDouble
			node.numSamples = linesIter.next.split("\t")(1).toLong
			node.weight = linesIter.next.split("\t")(1).toDouble
			val isLeaf : Boolean = linesIter.next.split("\t")(1).toBoolean
			if (!isLeaf) {
				node.featureId = linesIter.next.split("\t")(1).toInt
				node.featureType = linesIter.next.split("\t")(1).toInt
				if (node.featureType == 0) {
					// Continuous feature.
					node.threshold = linesIter.next.split("\t")(1).toDouble
				} else {
					// Discrete feature.
					node.leftValues = linesIter.next.split("\t")(1).split(",").map(_.toInt).toSet
					node.rightValues = linesIter.next.split("\t")(1).split(",").map(_.toInt).toSet
				}
				// Get children.
				val leftChild : Node = new Node
				val rightChild : Node = new Node
				leftChild.parent = Some(node)
				rightChild.parent = Some(node)
				node.leftChild = Some(leftChild)
				node.rightChild = Some(rightChild)
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
		rootNode
	}
	
	def readTree(nodesFile : String) : Node = {
		readTree(Source.fromFile(new File(nodesFile)).getLines)
	}
	
	
	// 6. Function for resetting the error statistics and train samples counts
	//    for nodes of a tree. Currently unused.
	
	def resetTreeStats(rootNode : Node) : Unit = {
		val nodesStack : Stack[Node] = Stack()
		nodesStack.push(rootNode)
		while (!nodesStack.isEmpty) {
			val node : Node = nodesStack.pop
			node.error = 0
			node.splitError = 0
			node.gain = 0
			node.numSamples = 0
			if (!node.isLeaf) {
				nodesStack.push(node.rightChild.get)
				nodesStack.push(node.leftChild.get)
			}
		}
	}

}
