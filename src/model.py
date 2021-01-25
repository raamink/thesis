from string import digits
from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers, Model
from collections import defaultdict, ChainMap

printInstantiation = True
printBuildInputs = False
class layerFactory:
    """General factory for creating layers."""
    # Expand factories with:
    # Keeps track of existing layer types that exist.
    layerTypes = {}
    layerID = str
    layerType = 'abstract'

    def __init__(self, *args, **kwargs):
        self.registerClass()
        if printInstantiation:
            print(f'instantiated a {self.layerType} factory')

    def registerClass(self):
        if self.layerType != layerFactory.layerType:
            layerFactory.layerTypes[self.layerType] = self

    def buildLayer(self, layerID, layerParms, architecture):
        if printBuildInputs: print(f'layerFactory: {layerType} * {layerParms}')
        self.layerTypes[layerID].buildLayer(layerParms, architecture)
        
    def buildBlock(self, blockType: str, blockParms: list, architecture: dict):
        if printBuildInputs: print('layerFactory: ', self.layerTypes[blockType], blockType, blockParms, architecture)
        self.layerTypes[blockType].buildBlock(blockParms, architecture)
    
class convFactory(layerFactory):
    layerType = 'Conv'

    def buildBlock(self, blockParms: list, architecture: dict) -> None:
        for layerName, layerParms in blockParms:
            inputID, _, parmsDict = layerParms
            inputTensor = architecture[inputID]
            newlayer = self.buildLayer(parmsDict, inputTensor)
            architecture[layerName] = newlayer

    def buildLayer(self, layerParms: dict, inputTensor: tf.Tensor) -> Callable:
        if printBuildInputs: print(f'> convFactory.buildLayer * layerParms: {layerParms}')
        bias = False if layerParms['batchNorm'] else True
        x = layers.Conv2D(filters= layerParms['filter'], kernel_size= layerParms['kernel'],
                strides= layerParms['stride'], padding=layerParms['padding'], 
                use_bias=bias)(inputTensor)
        
        if layerParms['batchNorm']:
            x = layers.BatchNormalization(epsilon=0.1)(x)
        if layerParms['leaky']:
            x = layers.LeakyReLU(alpha=0.1)(x)
        
        return x

class resnetFactory(layerFactory):
    layerType = 'resnet'
    
    def buildBlock(self, blockParms: list, architecture: dict) -> None:
        if printBuildInputs: print('resnet.buildBlock * ', blockParms)
        firstIn = None
        for layerName, layerParms in blockParms:
            inputID, _, parmsDict = layerParms
            inputTensor = architecture[inputID]
            if firstIn == None:
                firstIn = inputTensor
            newlayer = self.buildLayer(parmsDict, inputTensor)
            architecture[layerName] = newlayer
        architecture[layerName] = newlayer + firstIn
    
    def buildLayer(self, layerParms: dict, inputTensor: tf.Tensor) -> Callable:
        """
        Build function, which accepts a dictionary containing:
            - `kernel`, the convolutional kernel size 
            - The `stride`
            - The `filter`
            - `inputTensor`, which identifies the inputs.
        """
        if printBuildInputs: print(f'> convFactory.buildLayer * layerParms: {layerParms}')
        # Map dictionary entries to readable variables.
        stride, kernel, filterSize, bnorm, leak, padding = \
                (layerParms['stride'], layerParms['kernel'], layerParms['filter'], 
                layerParms['batchNorm'], layerParms['leaky'], layerParms['padding'])

        # derived names
        if stride > 1:
            x = layers.ZeroPadding2D(((1,0),(1,0)))(inputTensor)
            pad = 'valid'
        else:
            x = inputTensor
            pad = 'same'
        bias = False if bnorm else True
        pad = padding if padding else pad

        x = layers.Conv2D(filters=filterSize, kernel_size=kernel, 
                strides=stride, padding=pad, use_bias=bias)(x)
        
        if bnorm:
            x = layers.BatchNormalization(epsilon=0.1)(x)
        if leak:
            x = layers.LeakyReLU(alpha=0.1)(x)
        
        return x

        
        
        
    # def layerBlock(self, *, inputTensor=tf.Tensor, blockParms = None :dict, ):
    #     x = 
    #     for count, conv in enumerate(convs):
    #         if count == (len(convs)-1) and skip:
    #             skipConnection = x
    #         if conv['stride'] > 1: x = layers.ZeroPadding2D(((1,0),(1,0)))(x)
    #         x = layers.Conv2D(conv['filter'], conv['kernel'],strides=conv['stride'],
    #                 padding='valid' if conv['stride']>1 else 'same',
    #                 name=f"conv_{conv['layernum']}", 
    #                 use_bias=False if conv['bnorm'] else True)(x)
    #         if conv['bnorm']: x = layers.BatchNormalization(epsilon=0.001, 
    #                     name=f"bnorm_{conv['layernum']}")(x)
    #         if conv['leaky']: x = layers.LeakyReLU(alpha=0.1,
    #                     name=f"leaky_{conv['layernum']}")(x)
    #     return layers.merge.add([skipConnection, x] if skip else x)

class inputFactory(layerFactory):
    layerType = 'IN'
    
    def buildBlock(self, blockParms: list, architecture: dict):
        # print('inputFactory.buildBlock', '*', blockParms)
        for layerID, layerParms in blockParms:
            architecture[layerID] = self.buildLayer(layerParms)
        
    def buildLayer(self, layerParms: list):
        if printBuildInputs: print(f'> convFactory.buildLayer * layerParms: {layerParms}')
        # print('inputFactory.buildLayer', layerParms)
        _, _, parms = layerParms
        shape = parms['shape']
        return layers.Input(shape=shape)

class outputFactory(layerFactory):
    layerType = 'OUT'

    def buildBlock(self, *args):
        return self.buildLayer(*args)
    
    def buildLayer(self, layerParms, architecture):
        return layerParms['inputTensor']

class concatFactory(layerFactory):
    layerType = 'Concat'

    def buildBlock(self, blockParms: list, architecture: dict) -> None:
        for layerID, layerParms in blockParms:
            inputTensor = architecture[layerParms[0]]
            architecture[layerID] = self.buildLayer(layerParms, inputTensor, architecture)

    def buildLayer(self, layerParms: list, inputTensor: tf.Tensor, architecture: dict) -> Callable:
        _, _, parms = layerParms
        concats = [inputTensor]
        extras = parms['concatWith']
        if extras is list:
            concats += [architecture[extra] for extra in extras]
        else:
            concats.append(architecture[extras])
        return layers.concatenate(concats)

    
class myModel(Model):
    # def __init__(self, lossFunction, optimiser, trainLoss, trainMetric, 
            # testLoss, testMetric, architectureFile: str):

        # self.loss = lossFunction
        # self.optimiser = optimiser

        # self.trainLoss = trainLoss
        # self.testLoss  = testLoss
        
        # self.trainMetric = trainMetric
        # self.testMetric  = testMetric

    def __init__(self, architectureFile: str = None):
        super(myModel, self).__init__()

        self.collectBuilders()

        self.architecture = {}
        if architectureFile:
            self.buildArchitecture(architectureFile)


    def buildArchitecture(self, architectureFile):
        lines = buildLineIterator(architectureFile)
        blocks = buildBlockIterator(lines)

        for blockType, blockList in blocks:
            """There's no reason to shred the block apart at this level. Instead, direct
            blockList to correct blockBuilder, and recover the outputs of those blocks."""
            print(blockType, blockList)
            self.builders.buildBlock(blockType, blockList, self.architecture)

    
    def collectBuilders(self):
        self.builders = layerFactory()
        for subclass in layerFactory.__subclasses__():
            subclass()
                
def buildBlockIterator(lineIterator):
    """Generates an interator which returns layer blocks"""
    blockName = None
    blockParms = []
    for lineName, lineID, *lineParms in lineIterator:
        if lineName == blockName:
            blockParms.append([lineID, lineParms])
        elif blockName == None:
            blockName, blockParms = lineName, [[lineID, lineParms]]
        else:
            oldBlock = [blockName.rstrip(digits), blockParms]
            blockName, blockParms = lineName, [[lineID, lineParms]]
            yield oldBlock
    else:
        yield [lineName, [[lineID, lineParms]]]

def buildLineIterator(architectureFile):
    """Parses the layers and compiles dictionaries for them."""
    with open(architectureFile) as f:
        for line in f:
            if line == '\n' or line[0] == '#':
                continue
            
            blockName, layerID, layerInput, ops, *parms = line.strip().split(', ')
            parmsDict = {}            
            opParms = {}
            for parm in parms:
                k,v = parm.split('=')
                opParms[k] = v

            decoder = decode(blockName, ops)
            parmsDict = decoder(ops, opParms, parmsDict)

            yield blockName, layerID, layerInput, ops, parmsDict
            
def decode(blockName, ops):
    blockName = blockName.rstrip(digits)
    decoders = {'IN': decodeIn,
                'OUT': decodeIn,
                'Conv': decodeConv,
                'Concat': decodeConcat,
                'maxpool': decodeMaxPool}
    
    if blockName in decoders.keys():
        return decoders[blockName]
    elif blockName == 'resnet':
        firstOp, *_ = ops.split('+')
        return decoders[firstOp]
    else:
        raise ValueError(f'Block type "{blockName}"" not recognised')


def decodeIn(ops, opParms, parmsDict):
    defaults = {'shape': (None, None, 3)}
    remap = {'s':'shape'}
    for key,value in opParms.items():
        if key in remap.keys():
            parmsDict[remap[key]] = eval(value)
    
    return ChainMap(parmsDict, defaults)

def decodeConv(ops, opParms, parmsDict):
    """Labels the different parameters for convolutional blocks"""
    defaults = {'kernel' : 3, 'stride' : 1, 'filter' : 1, 'padding' : 'valid'}
    remap = {'k':'kernel', 's':'stride', 'p':'padding', 'f':'filter'}

    parmsDict = {remap[key]: eval(value) for (key,value) in opParms.items() if key in remap.keys()}
    parmsDict['batchNorm'] = True if 'BN' in ops else False
    parmsDict['leaky'] = True if 'ReLU' in ops else False
    
    return ChainMap(parmsDict, defaults)

def decodeConcat(ops, opParms, parmsDict):
    """Separates the input from the ops"""
    _, *extraConcats = ops.split(' ')
    if extraConcats:
        parmsDict['concatWith'] = extraConcats
        return parmsDict
    else:
        raise ValueError('Concatenates with nothing')

def decodeMaxPool(ops, opParms, parmsDict):
    defaults = {'pool_size': (2,2), 'strides': None}
    remap = {'ps':'pool_size', 's':'strides'}
    
    for parm in opParms:
        key, val = parm.split('=')
        if key not in remap.keys():
            continue
        val = eval(val)
        if type(val) == int:
            val = (val, val)
        parmsDict[remap[key]] = val
        
    return ChainMap(parmsDict, defaults)

if __name__ == "__main__":
    from pprint import pprint
    
    testModel = myModel('architectures/testArch.csv')
    

    print('ping')