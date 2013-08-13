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
	
}