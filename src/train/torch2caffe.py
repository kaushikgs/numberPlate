import caffe
import numpy as np
import sys
import subprocess

def readConvLine(line, numIn, kernelSize, outNo, weights):
    splitLine = line.split()
    c=0;
    for inNo in range(numIn):
        for rowNo in range(kernelSize):
            for colNo in range(kernelSize):
                weights[outNo,inNo,rowNo,colNo] = float(splitLine[c])
                c=c+1
    return weights

def readIPLine(line, numIn, outNo, weights):
    splitLine = line.split()
    c=0;
    for inNo in range(numIn):
        weights[outNo,inNo] = splitLine[c]
        c=c+1
    return weights

def readBiasLine(line, numOut):
    biases = np.zeros(numOut)
    splitLine = line.split()
    for i in range(numOut):
        biases[i] = float(splitLine[i])
    return biases

subprocess.call(['th', 'readTorch.lua', sys.argv[2], 'temp.txt'])

caffe.set_mode_gpu()
net = caffe.Net(sys.argv[1], caffe.TEST)

convKernels = [0,7,7,3,3,3,3,1,1,1,1]
featMaps = [3, 96,256,512,512,1024,9216,4096, 64, 2] #hack: during conversion from conv to ip
layerNo=0

with open('temp.txt') as tempFile:
    for line in tempFile:
        line = line.rstrip('\n')
        if not line:
            continue
        
        if line.startswith('conv') or line.startswith('ip'):
            print('layer num ' + str(layerNo) + ' successfuly converted')
            layerNo = layerNo+1 #index layers from 1
            splitLine = line.split()
            layerName = splitLine[0]
            numOut = int(splitLine[1])
            numIn = featMaps[layerNo-1]
            outNo = 0;
          
            if layerName.startswith('conv'):
                layerType = 'conv'
                weights = np.zeros((numOut, numIn, convKernels[layerNo], convKernels[layerNo]))
            elif layerName.startswith('ip'):
                layerType = 'ip'
                weights = np.zeros((numOut, numIn))
        
        elif outNo==numOut:
            np.copyto(net.params[layerName][1].data, readBiasLine(line, numOut))
        
        else:
            if layerType == 'conv':
                weights = readConvLine(line, numIn, convKernels[layerNo], outNo, weights)
            elif layerType == 'ip':
                weights = readIPLine(line, numIn, outNo, weights)
            if outNo == (numOut-1):
                np.copyto(net.params[layerName][0].data, weights, 'same_kind')
            outNo=outNo+1
      
net.save(sys.argv[3])
