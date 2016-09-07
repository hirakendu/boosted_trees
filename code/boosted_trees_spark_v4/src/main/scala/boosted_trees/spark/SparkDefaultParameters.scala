package boosted_trees.spark

object SparkDefaultParameters {

  // General.
  val sparkAppName: String = "Boosted Trees"
  val workDir: String = "file://" + System.getenv("HOME") + "/temp/boosted_trees_work"

  // SparkTrainTestSplitter.
  val dataFile: String = workDir + "/data.txt"
  val trainDataFile: String = workDir + "/split/train_data.txt"
  val testDataFile: String = workDir + "/split/test_data.txt"

  // SparkDataEncoder.
  val headerFile: String = workDir + "/header.txt"
  val dictsDir : String = workDir + "/encoding/dicts"
  val encodedTrainDataFile : String = workDir + "/encoding/encoded_train_data.txt"

  // SparkDecisionTreeModelTrainer.
  val treeModelDir: String = workDir + "/tree"

  // SparkGBDTModelTrainer.
  val forestModelDir: String = workDir + "/forest"

}
