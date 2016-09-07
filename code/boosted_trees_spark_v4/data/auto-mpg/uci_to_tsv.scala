#!/usr/bin/env scala

// Old header for Scala 2.9.
// #!/usr/bin/env scala
// exec scala "$0" "$@"
// !#

import java.io.File
import java.io.PrintWriter

import scala.io.Source

val data = Source.fromFile(new File("auto-mpg.data.txt")).
  getLines.toList.
  map(line => {
    val chars = line.toCharArray
    var inside = 0
    for (i <- 0 to chars.length - 1) {
      if (chars(i) == '"') {inside = 1 - inside; chars(i) = ' '}
      if ((chars(i) == ' ') && (inside == 1)) {chars(i) = '^'}
      if ((chars(i) == '\t') && (inside == 0)) {chars(i) = ' '}
    }
    chars.mkString
  }).
  map(_.split(" ").filter(!_.equals(""))).
  filter(_.length == 9).
  map(sample => {sample(8) = sample(8).split("\\^").filter(!_.equals(""))(0); sample}).
  map(_.mkString("\t").replaceAll("\t\\?", "\t0"))


var printWriter = new PrintWriter(new File("data.txt"))
printWriter.println(data.mkString("\n"))
printWriter.close

(new File("split/")).mkdirs
printWriter = new PrintWriter(new File("split/train_data.txt"))
printWriter.println(data.dropRight(99).mkString("\n"))
printWriter.close
printWriter = new PrintWriter(new File("split/test_data.txt"))
printWriter.println(data.drop(299).mkString("\n"))
printWriter.close
