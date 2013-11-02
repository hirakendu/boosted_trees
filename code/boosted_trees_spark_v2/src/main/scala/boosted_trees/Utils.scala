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

import scala.collection.mutable.{Set => MuSet}
import scala.util.Random
import scala.io.Source
import scala.util.Sorting

/**
 * Miscellaneous utility functions.
 */

object Utils {
	
	// 1. Function for taking a random combination n choose k.
	//    Useful for sampling a given number of samples without replacement
	//    from a given dataset.
	//    Algorithm is due to Robert Floyd from Jon Bentley's
	//    Programming Pearls column "A sample of Brilliance"
	//    via http://stackoverflow.com/questions/2394246/algorithm-to-select-a-single-random-combination-of-values
	
	def sampleWithoutReplacement(n : Long, k : Long) : Set[Long] = {
		var samples : MuSet[Long] = MuSet()
		val random : Random = new Random()
		for (i <- n - k to n - 1) {
			var t = random.nextLong()
			if (t < 0) {
				t = -t
			}
			t = t % (i + 1)
			if (!samples.contains(t)) {
				samples.add(t)
			} else {
				samples.add(i)
			}
		}
		return samples.toSet
	}
	
	
	// 2. Functions for printing list of doubles.
	
	// 2.1. Print list of doubles to a small precision.
	
	def mkStringDouble(v : Iterable[Double], delimiter : String) : String = {
		var str = ""
		v match {
			case Nil => str
			case _ : Iterable[Double] => for (e <- v.zipWithIndex) {
				if (e._2 != 0) {
					str += delimiter
				}
				str += "%.3f".format(e._1)
			}
		}
		str
	}
	
	def mkStringDouble(v : Iterable[Double]) : String = {
		mkStringDouble(v, ", ")
	}
	
	// 2.2. Print list of doubles to high precision.
	
	def mkStringDoublePrecise(v : Iterable[Double], delimiter : String) = {
		var str = ""
		v match {
			case Nil => str
			case _ : Iterable[Double] => for (e <- v.zipWithIndex) {
				if (e._2 != 0) {
					str += delimiter
				}
				str += "%.6f".format(e._1)
			}
		}
		str
	}
	
	
	// 3. Function for creating base directory for a path.
	
	def createParentDirs(pathName : String) : Unit = {
		val pathNameNoSlash : String = pathName.replaceAll("/+$", "")
		val parts : Array[String] = pathNameNoSlash.split("/")
		val dirName : String = pathNameNoSlash.replaceAll(parts(parts.length - 1) + "$", "")
		(new File(dirName)).mkdirs
	}

	
	// 4. Function for computing ROC and AUC.
	
	def findRocAuc(scoresLabels : Array[(Double, Int)]) : (Array[(Double, Double, Double)], Double) = {
		val sortedScoresLabels : Array[(Double, Int)] = scoresLabels.sortWith(_._2 < _._2)
		Sorting.stableSort(sortedScoresLabels, (x : (Double, Int), y : (Double, Int)) => (x._1 > y._1))
		val numZeros : Double = scoresLabels.filter(_._2 == 0).length
		val numOnes : Double = scoresLabels.filter(_._2 == 1).length
		if (numZeros == 0) {
			return (Array((0.0, 0.0, sortedScoresLabels(0)._1 + 0.1),
					(0.0, 1.0, sortedScoresLabels(sortedScoresLabels.length - 1)._1 - 0.1),
					(1.0, 1.0, sortedScoresLabels(sortedScoresLabels.length - 1)._1 - 0.1)), 1.0)
		}
		if (numOnes == 0) {
			return (Array((0.0, 0.0, sortedScoresLabels(0)._1 + 0.1),
					(0.0, 1.0, sortedScoresLabels(0)._1 + 0.1),
					(1.0, 1.0, sortedScoresLabels(sortedScoresLabels.length - 1)._1 - 0.1)), 1.0)
		}
		val rocPoints : Array[(Double, Double, Double)] = new Array(numZeros.toInt + 2)
		rocPoints(0) = (0, 0, sortedScoresLabels(0)._1 + 0.1)
		rocPoints(numZeros.toInt + 1) = (1, 1, sortedScoresLabels(sortedScoresLabels.length - 1)._1 - 0.1)
		var truePositives : Int = 0
		var falsePositives : Int = 0
		for (scoreLabel <- sortedScoresLabels) {
			if (scoreLabel._2 == 1) {
				truePositives += 1
			} else if (scoreLabel._2 == 0) {
				rocPoints(falsePositives + 1) = (falsePositives / numZeros,
						truePositives / numOnes, scoreLabel._1)
				falsePositives += 1
			}
		}
		var auc : Double = rocPoints.map(_._2).drop(1).reduce(_ + _) / numZeros
		(rocPoints, auc)
	}

}
