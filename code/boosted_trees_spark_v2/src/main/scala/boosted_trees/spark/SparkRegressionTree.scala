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

import java.io.PrintWriter
import java.io.StringWriter

import scala.collection.mutable.{Set => MuSet}
import scala.collection.mutable.{Map => MuMap}
import scala.collection.mutable.MutableList
import scala.collection.mutable.Stack
import scala.collection.parallel.immutable.ParSeq
import scala.util.control.Breaks._

import spark.RDD
import spark.SparkContext
import spark.SparkContext._

import boosted_trees.Utils
import boosted_trees.Node
import boosted_trees.RegressionTree

/**
 * Distributed versions of functions in boosted_trees.RegressionTree
 * to run on Spark and read/write using Hadoop.
 */

object SparkRegressionTree {
	
	// 1. Functions for training a tree model.
	
	// 1.1. Function for training a single node, i.e., finding best split
	//      for a given dataset.
	
	def trainNode(node : Node, samples : RDD[Array[Double]],
			featureTypes : Array[Int], featureWeights : Array[Double],
			maxDepth : Int = 4, minGain : Double = 1e-6) :
			(RDD[Array[Double]], RDD[Array[Double]]) = {
		
		// 0. Parameters.
		val maxNumQuantileValues : Int = 1000  // Q-1. At least 1, i.e., Q >= 2.
		val maxNumQuantileSamples : Int = 10000
		
		// 1. Some node statistics, prior to training.
		println("        Calculating initial node statistics.")
		var initialTime : Long = System.currentTimeMillis
		
		val stats : (Long, Double, Double) = samples.
				map(sample => (1L, sample(0), sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
		node.numSamples = stats._1
		node.response = stats._2 / stats._1
		node.error = stats._3 - stats._2 * stats._2 / stats._1
		
		var finalTime : Long = System.currentTimeMillis
		println("        Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
				
		// 2. Don't split if not enough samples or depth too high
		//    or no variable to split.
		//    or other due to other termination criteria.
		//    Additional criteria are used to stop the split further along.
		if (node.numSamples < 2) {
			return (null, null)
		}
		if (node.id >= Math.pow(2, maxDepth)) {
			return (null, null)
		}
		
		// 3. Find best split for each feature.
		val numSamples : Long = node.numSamples
		val numFeatures : Int = featureTypes.length
		val errorsForFeatures : Array[Double] = new Array(numFeatures)
		val thresholdsForFeatures : Array[Double] = new Array(numFeatures)
		val leftValuesForFeatures : Array[Set[Int]]  = new Array(numFeatures)
		val rightValuesForFeatures : Array[Set[Int]]  = new Array(numFeatures)
		
		// 3.1. Find the candidate thresholds for continuous features.
		println("        Calculating quantiles/candidate thresholds for each continuous feature.")
		initialTime = System.currentTimeMillis
		val candidateThresholdsForFeatures : Array[Array[Double]] = new Array(numFeatures)
		val numQuantileSamples : Int = Math.min(maxNumQuantileSamples.toLong, node.numSamples).toInt
		val numQuantileValues : Int = Math.min(maxNumQuantileValues, numQuantileSamples - 1)
		val quantileSamples : List[Array[Double]] = samples.takeSample(false, numQuantileSamples, 42).toList
		// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		for (j <- 1 to numFeatures - 1) {
			// Skip i = 0 corresponding to label.
			if (featureTypes(j) == 0) {
				val featureValues : List[Double] = quantileSamples.map(_(j)).sort(_ < _)
				val candidateThresholdsSet : MuSet[Double] = MuSet()
				for (q <- 1 to numQuantileValues) {
					val id : Int = (q * numQuantileSamples / (numQuantileValues + 1.0) - 1).toInt
					if (id >= 0) {
						val threshold : Double = (featureValues(id) + featureValues(id + 1)) / 2.0
						candidateThresholdsSet.add(threshold)
					}
				}
				candidateThresholdsForFeatures(j) =
					candidateThresholdsSet.toList.sort(_ < _).toArray
			}
		}
		// })  // End of j-loop for finding candidate thresholds for continuous features.
		finalTime = System.currentTimeMillis
		println("        Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		
		println("        Calculating histograms for each feature.")
		initialTime = System.currentTimeMillis
		
		// 3.2. Find the stats (histogram, mean response, mean square response) 
		//      for various feauture-value bins. Most data intensive.
		// Method 1: giant reduceByKey, potentially doing smaller reduces on partitions implicitly.
//		val statsForFeatureBins : List[((Int, Int), (Long, Double, Double))] =
//			samples.flatMap(sample => {
//				val square : Double = sample(0) * sample(0)
//				val valueBinsForFeatures : Array[Int] = new Array(numFeatures)
//				// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
//				for (j <- 1 to numFeatures - 1) {
//					
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
//					map(featureBin => (featureBin, (1L, sample(0), square)))
//			}).
//			reduceByKey((stats1, stats2) => (stats1._1 + stats2._1,
//						stats1._2 + stats2._2, stats1._3 + stats2._3)).
//			collect.toList
			
		// Method 2: iterative reduces on mapPartitions + final reduce.
		val statsForFeatureBins : List[((Int, Int), (Long, Double, Double))] =
			samples.mapPartitions(samplesIterator => {
				val statsForFeatureBinsMap : MuMap[(Int, Int), (Long, Double, Double)] = MuMap()
				while (samplesIterator.hasNext) {
					val sample : Array[Double] = samplesIterator.next
					val square : Double = sample(0) * sample(0)
					// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
					for (j <- 1 to numFeatures - 1) {	
						var bj : Int = 0  // Bin index of the value.
						if (featureTypes(j) == 0) {
							// Continuous feature.
							val numBins : Int =  candidateThresholdsForFeatures(j).length + 1
							bj = numBins - 1
							breakable { for (b <- 0 to numBins - 2) {
								if (sample(j) < candidateThresholdsForFeatures(j)(b)) {
									bj = b
									break
								}
							} }
						} else if (featureTypes(j) == 1) {
							// Discrete feature.
							bj = sample(j).toInt
						}
						if (statsForFeatureBinsMap.contains((j, bj))) {
							val oldStats : (Long, Double, Double) = statsForFeatureBinsMap(j, bj)
							statsForFeatureBinsMap((j, bj)) = (oldStats._1 + 1, oldStats._2 + sample(0),
									oldStats._3 + square)
						} else {
							statsForFeatureBinsMap((j, bj)) = (1L, sample(0), square)
						}
					}
					// })
				}
				Iterator(statsForFeatureBinsMap)
			}).
			reduce((map1, map2) => {
				val merged : MuMap[(Int, Int), (Long, Double, Double)] = map1.clone
				for (key <- map2.keySet) {
					if (merged.contains(key)) {
						merged(key) = (merged(key)._1 + map2(key)._1,
								merged(key)._2 + map2(key)._2,
								merged(key)._3 + map2(key)._3)
					} else {
						merged(key) = map2(key)
					}
				}
				merged
			}).toList
		
//		// Method 2b: same as 2, use aggregate and combiner functions instead of mapPartitions.
//		def statsAggregator(statsForFeatureBinsMap : MuMap[(Int, Int), (Long, Double, Double)],
//					sample : Array[Double]) : MuMap[(Int, Int), (Long, Double, Double)] = {
//			val square : Double = sample(0) * sample(0)
//			// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
//			for (j <- 1 to numFeatures - 1) {
//				var bj : Int = 0  // Bin index of the value.
//				if (featureTypes(j) == 0) {
//					// Continuous feature.
//					val numBins : Int =  candidateThresholdsForFeatures(j).length + 1
//					bj = numBins - 1
//					breakable { for (b <- 0 to numBins - 2) {
//						if (sample(j) < candidateThresholdsForFeatures(j)(b)) {
//							bj = b
//							break
//						}
//					} }
//				} else if (featureTypes(j) == 1) {
//					// Discrete feature.
//					bj = sample(j).toInt
//				}
//				if (statsForFeatureBinsMap.contains((j, bj))) {
//					val oldStats : (Long, Double, Double) = statsForFeatureBinsMap(j, bj)
//					statsForFeatureBinsMap((j, bj)) = (oldStats._1 + 1, oldStats._2 + sample(0),
//							oldStats._3 + square)
//				} else {
//					statsForFeatureBinsMap((j, bj)) = (1L, sample(0), square)
//				}
//			}
//			// })
//			statsForFeatureBinsMap
//		}
//		def statsReducer(map1 :  MuMap[(Int, Int), (Long, Double, Double)],
//				map2 :  MuMap[(Int, Int), (Long, Double, Double)]) :
//				MuMap[(Int, Int), (Long, Double, Double)] = {
//			val merged : MuMap[(Int, Int), (Long, Double, Double)] = map1.clone
//			for (key <- map2.keySet) {
//				if (merged.contains(key)) {
//					merged(key) = (merged(key)._1 + map2(key)._1,
//							merged(key)._2 + map2(key)._2,
//							merged(key)._3 + map2(key)._3)
//				} else {
//					merged(key) = map2(key)
//				}
//			}
//			merged
//		}
//		val emptyStatsMap : MuMap[(Int, Int), (Long, Double, Double)] = MuMap()
//		val statsForFeatureBins : List[((Int, Int), (Long, Double, Double))] =
//			samples.aggregate(emptyStatsMap)(statsAggregator, statsReducer).toList
		
		// 3.3. Separate the histograms, means, and calculate errors.
		//      Sort bins of discrete features by mean responses.
		//		Note that we can read in parallel from
		//		statsForFeatureBins.
		val statsForBinsForFeatures: Array[List[(Int, Long, Double, Double)]] = new Array(numFeatures)
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			val statsForBins : List[(Int, Long, Double, Double)] =
				statsForFeatureBins.filter(_._1._1 == j).
					map(statsForFeatureBin => (statsForFeatureBin._1._2,
							statsForFeatureBin._2._1,
							statsForFeatureBin._2._2,
							statsForFeatureBin._2._3))
			if (featureTypes(j) == 0) {
				// For continuous features, order by bin indices,
				// i.e., order of quantiles.
				statsForBinsForFeatures(j) = statsForBins.sort(_._1 < _._1)
			} else if (featureTypes(j) == 1) {
				// For categorical features, order by means of bins,
				// i.e., order of means of values.
				statsForBinsForFeatures(j) = statsForBins.
						sort((stats1, stats2) => stats1._3/stats1._2 < stats2._3/stats2._2)
			}
		// }
		})
		
		// Done calculating histograms.
		finalTime = System.currentTimeMillis
		println("        Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		
		println("        Finding best split and error.")
		initialTime = System.currentTimeMillis
		
		// 3.4. Calculate the best split and error for each feature.
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			
			val statsForBins : List[(Int, Long, Double, Double)] = statsForBinsForFeatures(j)
			val numBins : Int = statsForBins.length
			
			// Initial split and error.
			var leftNumSamples : Long =  statsForBins(0)._2
			var leftSumResponses : Double = statsForBins(0)._3
			var leftSumSquares : Double = statsForBins(0)._4
			var rightNumSamples : Long = 0
			var rightSumResponses : Double = 0
			var rightSumSquares : Double = 0
			for (b <- 1 to numBins - 1) {
				rightNumSamples += statsForBins(b)._2
				rightSumResponses += statsForBins(b)._3
				rightSumSquares += statsForBins(b)._4
			}
			var bMin : Int = 0
			var minError : Double = 0
			if (leftNumSamples != 0) {
				minError += leftSumSquares - leftSumResponses * leftSumResponses / leftNumSamples
			}
			if (rightNumSamples != 0) {
				minError += rightSumSquares - rightSumResponses * rightSumResponses / rightNumSamples
			}
			
			// Slide threshold from left to right and find best threshold.
			for (b <- 1 to numBins - 2) {
				leftNumSamples +=  statsForBins(b)._2
				leftSumResponses += statsForBins(b)._3
				leftSumSquares += statsForBins(b)._4
				rightNumSamples -= statsForBins(b)._2
				rightSumResponses -= statsForBins(b)._3
				rightSumSquares -= statsForBins(b)._4
				var error : Double = 0
				if (leftNumSamples != 0) {
					error += leftSumSquares - leftSumResponses * leftSumResponses / leftNumSamples
				}
				if (rightNumSamples != 0) {
					error += rightSumSquares - rightSumResponses * rightSumResponses / rightNumSamples
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
		var jMax : Int = 1
		var maxWeightedGain : Double = (node.error - errorsForFeatures(1)) * featureWeights(1)
		for (j <- 2 to numFeatures - 1) {
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
		
		// Done finding best split and error.
		finalTime = System.currentTimeMillis
		println("        Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		
		// 3.6. Don't split if no gain.
		if (node.gain <= minGain) {
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
		var leftSamples : RDD[Array[Double]] = null
		var rightSamples : RDD[Array[Double]] = null
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
	
	def trainTree(samples: RDD[Array[Double]], featureTypes : Array[Int],
			featureWeights : Array[Double],
			maxDepth : Int = 4, minGainFraction : Double = 0.01,
			minDistributedSamples : Int = 10000) : Node = {
		val rootNode : Node = new Node
		rootNode.id = 1
		rootNode.depth = 0
		// Find initial error to determine minGain.
		val stats : (Long, Double, Double) = samples.
				map(sample => (1L, sample(0), sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
		val minGain : Double = minGainFraction * (stats._3 - stats._2 * stats._2 / stats._1)
		// Option 1: Recursive method may cause stack overflow.
		// findBestSplitRecursive(rootNode)
		// Option 2: Iterative by maintaining a queue.
		var nodesStack : Stack[(Node,  RDD[Array[Double]])] = Stack()
		nodesStack.push((rootNode, samples))
		while (!nodesStack.isEmpty) {
			val initialTime : Long = System.currentTimeMillis
			var (node, nodeSamples) : (Node,  RDD[Array[Double]]) = nodesStack.pop
			if (nodeSamples.count >= minDistributedSamples) {
				println("      Training Node # " + node.id + ".")
				val (leftSamples, rightSamples) :
					(RDD[Array[Double]], RDD[Array[Double]]) =
						trainNode(node, nodeSamples, featureTypes, featureWeights, maxDepth, minGain)
				if (!node.isLeaf) {
					nodesStack.push((node.rightChild.get, rightSamples))
					nodesStack.push((node.leftChild.get, leftSamples))
				}
			} else {
				println("      Training Node # " + node.id + " (local).")
				val (leftSamplesList, rightSamplesList) :
					(List[Array[Double]], List[Array[Double]]) =
						RegressionTree.trainNode(node, nodeSamples.collect.toList,
								featureTypes, featureWeights, maxDepth, minGain)
				if (!node.isLeaf) {
					val leftSamples : RDD[Array[Double]] = samples.context.parallelize(leftSamplesList, 1)
					val rightSamples : RDD[Array[Double]] = samples.context.parallelize(rightSamplesList, 1)
					nodesStack.push((node.rightChild.get, rightSamples))
					nodesStack.push((node.leftChild.get, leftSamples))
				}
			}
			val finalTime : Long = System.currentTimeMillis
			println("      Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		}
		rootNode
	}
	
	
	// 2. Functions for saving and printing a tree model.
	
	// 2.1. Function to save a tree model in text format for later use.
	
	def saveTree(sc : SparkContext, nodesFile : String, rootNode : Node) : Unit = {
		val treeModelText : List[String] = RegressionTree.saveTree(rootNode)
		sc.parallelize(treeModelText, 1).saveAsTextFile(nodesFile)
	}
	
	// 2.2. Function to print a tree model for easy reading.
	
	def printTree(sc : SparkContext, treeFile : String,  rootNode : Node) : Unit = {
		sc.parallelize(RegressionTree.printTree(rootNode), 1).saveAsTextFile(treeFile)
	}
	
	
	// 3. Function for reading a tree model generated by *RegressionTree.
	
	def readTree(sc : SparkContext, nodesFile : String) : Node = {
		RegressionTree.readTree(SparkUtils.readSmallFile(sc, nodesFile).toIterator)
	}
	
}