package boosted_trees.local

object DefaultParameters {

  // General.
  val workDir: String = System.getenv("HOME") + "/temp/boosted_trees_work"

  // TrainTestSplitter.
  val dataFile: String = workDir + "/data.txt"
  val trainDataFile: String = workDir + "/split/train_data.txt"
  val testDataFile: String = workDir + "/split/test_data.txt"

  // DataEncoder.
  val headerFile: String = workDir + "/header.txt"
  val dictsDir : String = workDir + "/encoding/dicts/"
  val encodedTrainDataFile : String = workDir + "/encoding/encoded_train_data.txt"

  // DecisionTreeModelTrainer.
  val treeModelDir: String = workDir + "/tree/"

  // GBDTModelTrainer.
  val forestModelDir: String = workDir + "/forest/"

}
