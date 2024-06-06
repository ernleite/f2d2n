package com.deeplearning.layer

import com.deeplearning.Network.{generateRandomFloat}
import com.deeplearning.{ComputeActivation, CostManager, Network}
import com.deeplearning.samples.{CifarData, MnistData, TrainingDataSet}

import java.time.Instant

class DenseInputLayer extends InputLayer {
  var parameterSended = false
  var nablas_w_tmp = Array.empty[Float]

  private var dataSet: TrainingDataSet = if (Network.trainingSample == "Mnist") {
    println("Loading Dataset MINST")
    dataSet = new MnistData()
    dataSet
  } else {
    dataSet = new CifarData()
    dataSet
  }
  def computeInputWeights(epoch:Int, correlationId: String, yLabel:Int, startIndex:Int, endIndex:Int, index:Int, layer:Int, internalSubLayer:Int,params: scala.collection.mutable.HashMap[String,String]): Array[Array[Float]] = {
    if (lastEpoch != epoch) {
      counterTraining = 0
      counterBackPropagation = 0
      counterFeedForward = 0
      lastEpoch = epoch
    }
    epochCounter = epoch
    counterTraining += 1
    val startedAt = Instant.now
    val nextLayer = layer+1
    if (!wInitialized) {
      val arraySize = Network.InputLayerDim
      nabla_w = Array.ofDim(arraySize)
      dataSet.loadTrainDataset(Network.InputLoadMode)

      //weights = heInitialization(Network.getHiddenLayers(nextLayer), endIndex-startIndex, 1)
      //weights = generateRandomFloat(LayerManager.GetDenseActivationLayerStep(nextLayer)*(endIndex-startIndex))
      weights = generateRandomFloat(Network.getHiddenLayers(nextLayer, "hidden")*(endIndex-startIndex))
      //parameters += ("min" -> "0")
      //parameters += ("max" -> "0")
      //parameters += ("weighted_min" -> "0")
      //parameters += ("weighted_max" -> "0")
      wInitialized = true
    }

    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      val input = dataSet.getTrainingInput(index)
      val x = input.slice(startIndex + 2, endIndex + 2) //normalisation
      val v = CostManager.divide(x, 255)
      this.X += (correlationId -> v)
    }

    var receiverUCs = Network.getHiddenLayers(1, "hidden")

    //val x = Array.fill(LayerManager.GetDenseActivationLayerStep(nextLayer))(this.X(correlationId)).flatten
    val x = Array.fill(receiverUCs)(this.X(correlationId)).flatten

    val w1 = CostManager.dotProduct(weights, x)
    //split the array to be sent to layer +1 UCs
    val splittedLayerDim = Network.HiddenLayersDim(layer)
    val w2 = w1.grouped(w1.size/splittedLayerDim).toArray

    if (!parameterSended) {
    //  parameters("min") = weights.min.toString
    //  parameters("max") = weights.max.toString
      parameterSended = true
    }

    weighted(correlationId) = w1
//    val splitWeights = X(correlationId).grouped(receiverUCs).toArray
    val l = this.X(correlationId).length

    for (i: Int <- 0 until Network.getHiddenLayersDim(nextLayer, "hidden")) {
      val arr = w2(i)
      val weighted = arr.grouped(l).toArray.map(_.sum)

      // forward stats infos
      if (Network.CheckNaN) {
        val test = arr.zipWithIndex.filter { case (value, _) => value.isNaN }
        if (test.nonEmpty) {
          println("NaN values found at indices:")
        } else {
          println("No NaN values found in the array.")
        }
      }
      val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + i)
      actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, correlationId, yLabel, Network.MiniBatchRange, weighted, i, nextLayer, Network.InputLayerDim, params, Array.empty[Float])
    }
    w2
  }

  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Float, internalSubLayer: Int, fromInternalSubLayer: Int, params: scala.collection.mutable.HashMap[String,String]): Boolean = {
    counterBackPropagation += 1
    minibatch(correlationId) += 1
    //params("eventBP") =  (params("eventBP").toInt + 1).toString

    if (!backPropagateReceived.contains(correlationId)) {
      val fromArraySize = Network.getHiddenLayersDim(1, "hidden")
      backPropagateReceived += (correlationId -> true)
      //nablas_w(correlationId) = Array.ofDim(fromArraySize)
    }
    var startedAt = Instant.now
    if (nablas_w_tmp.isEmpty)
      nablas_w_tmp =  CostManager.dotProduct3(delta.length, delta, X(correlationId)).flatten
    else
      nablas_w_tmp = CostManager.sum2(nablas_w_tmp, CostManager.dotProduct3(delta.length, delta, X(correlationId)).flatten)
    //nablas_w(correlationId) =CostManager.dotProduct3(delta.length, delta, X(correlationId)).flatten
    //////////////////////////nablas_w(correlationId)(fromInternalSubLayer) = dot
    // how many calls would we received
    val callerSize = Network.getHiddenLayersDim(1, "hidden")

    // check if we reach the last mini-bacth
    //context.log.info("Receiving from bakpropagation")
    if ((Network.MiniBatch * callerSize) == minibatch.values.sum) {
      val a = Network.rangeInitStart
      val b = Network.rangeInitEnd
      parameterSended = false
      params("min") = "0"
      params("max") = "0"

      /*
      val flatten = nablas_w.values.toArray
      val reduce = flatten.transpose.map(_.sum)
      val nablaflaten = reduce

       */
      val tmp2 = CostManager.matMulScalar(learningRate / Network.MiniBatch,nablas_w_tmp)
      val tmp1 = CostManager.matMulScalar(1 - learningRate * (regularisation / nInputs), weights)
      weights = CostManager.minus2(tmp1, tmp2)

      //val scalling = CostManager.scalling(weights, Network.InputLayerDim, parameters("weighted_min").toFloat, parameters("weighted_min").toFloat)
      //weights = scalling

      //println("---------------------------------------------------------------")
      //println("Back-propagation event IL (0): Weights updated")

      backPropagateReceived.clear()
      minibatch.clear()
      weighted.clear()
      nablas_w.clear()
      counterBackPropagation=0
      nablas_w_tmp = Array.empty[Float]

      this.X.clear()
      true
    }
    else
      false
  }

  def FeedForwardTest(correlationId: String, startIndex: Int, endIndex: Int, index: Int, internalSubLayer: Int, layer: Int):  Array[Array[Float]] = {
    if (!wTest) {
      wTest = true
      //Temporary
      // read shard data from data lake
      dataSet.loadTestDataset(Network.InputLoadMode)
    }

    val nextLayer = layer + 1
    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      //for (i <- index until (index+1)) {
      val input = dataSet.getTestInput(index)
      val x = input.slice(startIndex + 2, endIndex + 2) //normalisation
      val v = CostManager.divide(x, 255)
      this.XTest += (correlationId -> v)
      wTest = true
    }

    val x = Array.fill(Network.getHiddenLayers(1, "hidden"))(this.XTest(correlationId)).flatten
    val w1 = CostManager.dotProduct(weights, x)
    //split the array to be sent to layer +1 UCs
    val splittedLayerDim = Network.HiddenLayersDim(layer)
    val w2 = w1.grouped(w1.size / splittedLayerDim).toArray
    for (i: Int <- 0 until Network.getHiddenLayersDim(nextLayer, "hidden")) {
      val arr = w2(i)
      val l = this.XTest(correlationId).length
      val weighted = arr.grouped(l).toArray.map(_.sum)
      val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + i)
      actorHiddenLayer ! ComputeActivation.FeedForwardTest(correlationId, weighted, i, nextLayer, Network.InputLayerDim)
    }

    weighted -= (correlationId)
    minibatch -= (correlationId)
    XTest -= (correlationId)
    w2
  }
}
