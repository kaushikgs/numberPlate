import caffe
import numpy as np
import sys
import lutorpy as lua
from operator import mul
require("cunn")
caffe.set_mode_gpu()

def getNextLayer(torchCNN, currentLayer):
    currentLayer = currentLayer+1
    while currentLayer < len(torchCNN.modules):
        if(torchCNN.modules[currentLayer].weight):
            return currentLayer
        currentLayer = currentLayer+1
    return -1

def isCompatible(torchShape, caffeShape):
    if torchShape[0] != caffeShape[0]:
        return False

    if len(caffeShape) == 1:
        return True

    torchNum = reduce(mul, torchShape[1:])
    caffeNum = reduce(mul, caffeShape[1:])

    if(torchNum == caffeNum):
        return True
    else:
        return False

caffeCNN = caffe.Net(sys.argv[1], caffe.TEST)
torchCNN = torch.load(sys.argv[2])

torchLayerNum = -1
for layerName in caffeCNN.params:
    torchLayerNum = getNextLayer(torchCNN, torchLayerNum)
    torchWeights = torchCNN.modules[torchLayerNum].weight.asNumpyArray()
    torchBiases = torchCNN.modules[torchLayerNum].bias.asNumpyArray()

    if not isCompatible (torchWeights.shape, caffeCNN.params[layerName][0].data.shape):
        sys.exit('Incompatible torch network and caffe model')

    caffeWeightsShape = caffeCNN.params[layerName][0].data.shape
    caffeWeights = torchWeights.reshape( caffeWeightsShape)
    caffeBiases = torchBiases

    np.copyto(caffeCNN.params[layerName][0].data, caffeWeights)
    np.copyto(caffeCNN.params[layerName][1].data, caffeBiases)

caffeCNN.save(sys.argv[3])