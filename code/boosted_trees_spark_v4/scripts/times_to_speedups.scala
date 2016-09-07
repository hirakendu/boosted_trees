#!/usr/bin/env scala

// Old header for Scala 2.9.
// #!/usr/bin/env scala
// exec scala "$0" "$@"
// !#

import java.io.File
import java.io.PrintWriter

import scala.io.Source


var iBase = 0  // baseline
if (args.length >= 3) {
  iBase = args(2).toInt - 1
}

val times = Source.fromFile(new File(args(0))).getLines.toArray.
    map(_.split("\t", -1).map(_.toDouble))
val speedups = Array.ofDim[Double](times.length, times(0).length)
for (i <- 0 to times.length - 1) {
  for (j <- 0 to times(i).length - 1) {
    speedups(i)(j) = times(iBase)(j) / times(i)(j)
  }
}

val printWriter = new PrintWriter(new File(args(1)))
printWriter.println(speedups.map(_.map("%.5f".format(_)).mkString("\t")).mkString("\n"))
printWriter.close
