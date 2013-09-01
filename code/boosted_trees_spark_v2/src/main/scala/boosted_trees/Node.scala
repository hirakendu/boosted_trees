package boosted_trees

class Node extends Serializable {
	
	var id : Int = 1
	var depth : Int = 0
	
	var response : Double = 0.0

	var numSamples : Long = 0
	var weight : Double = 0.0
	var error : Double = 0.0
	var splitError : Double = 0.0
	var gain : Double = 0.0
	
	var featureId : Int = -1
	var featureType : Int = -1
	var leftValues : Set[Int] = Set()
	var rightValues : Set[Int] = Set()
	var threshold : Double = 0.0
	
	// Left = 0, right = 1.
	var leftChild : Option[Node] = None
	var rightChild : Option[Node] = None
	
	var parent : Option[Node] = None
	
	def isLeaf() : Boolean = {
		if (leftChild.isEmpty && rightChild.isEmpty) {
			return true
		}
		return false
	}
	
}
