import tensorflow as tf
from tensorflow.keras import layers, Model
from collections import defaultdict, ChainMap

class layerFactory:
    """General factory for creating layers."""
    # Expand factories with:
    # Keeps track of existing layer types that exist.
    layerTypes = {}
    layerID = str

    def __init__(self, layerID, cellCount, *args, **kwargs):
        self.registerClass()
        print('instantiated a general factory')

    @classmethod
    def registerClass(cls, layerType):
        layerFactory.layerTypes[cls.layerType] = cls.__qualname__

    def buildLayer(*args, **kwargs):
        pass

class ConvFactory(layerFactory):
    layerType = 'conv2D'
    
    def buildLayer(self, layerParms):
        """
        Build function, which accepts a dictionary containing:
            - `kernel`, the convolutional kernel size 
            - The `stride`
            - The `filter`
            - `inputTensor`, which identifies the inputs.
        """
        #Fills out missing parms, then maps to aliases
        defaults = {'batchNorm' = True, 'leaky' = True, 'skip' = True}
        parms = ChainMap(layerParms,defaults)

        stride, kernel, filterSize, inputTensor, bnorm, leak, skip = 
                (parms['stride'], parms['kernel'], parms['filter'], 
                parms['inputTensor'], parms['batchNorm'], parms['leaky'],
                parms['skip'], parms['layerNum'])

        if stride > 1:
            x = layers.ZeroPadding2D(((1,0),(1,0)))(inputTensor)
            pad = 'valid'
        else:
            x = inputTensor
            pad = 'same'

        bias = False if bnorm else True

        x = layers.Conv2D(filters=filterSize, kernel_size=kernel, 
                strides=stride, padding=pad, name=f'conv_{layerNum}',
                use_bias=bias)(x)
        
        if bnorm:
            x = layers.BatchNormalization(epsilon=0.1, 
                    name=f'bnorm_{layerNum}')(x)
        if leak:
            x = layers.LeakyReLU(alpha=0.1, name=f'leaky_{layerNum}')(x)
        
        return x
    
    def buildBlock(self, blockSize, *):
        pass
        
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
        
class model(Model):
    def __init__(self, lossFunction, optimiser, trainLoss, trainMetric, 
            testLoss, testMetric, architectureFile: str):
        super(model, self).__init__()

        self.loss = lossFunction
        self.optimiser = optimiser

        self.trainLoss = trainLoss
        self.testLoss  = testLoss
        
        self.trainMetric = trainMetric
        self.testMetric  = testMetric

        # Placeholders for input and output tensors.
        self.input = tf.Variable()
        self.output = tf.Variable()

        self.architecture = self.buildArchitecture(architectureFile)


    def buildArchitecture(self, architectureFile)
        blockIterator = buildBlockIterator(architectureFile)
        builtLayers = dict()
        output = None

        for block in blockIterator:
            layerIterator = iter(block)    
            for layerType, layerNum, layerInputs, layerParms in layerIterator:
                inputLayer = builtLayers[layerInputs]
                builtLayers[layerNum] = builders[layerType].buildLayer(
                        inputTensor = inputLayer, layerNum=layerNum, **layerParms)
                builtLayers[layerNum].printBuild()
        return builtLayers
    
                
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
            oldBlock = [blockName, blockParms]
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
            parmsDict = defaultdict()
            parmsDict['inputTensor'] = layerInput
            opParms = {}
            for parm in parms:
                k,v = parm.split('=')
                opParms[k] = v

            if blockName == 'Concat':
                parmsDict = decodeConcat(ops, opParms, parmsDict)
            elif 'Conv' in blockName:
                parmsDict = decodeConv(ops, opParms, parmsDict)
            elif blockName in ['IN', 'OUT']:
                ops = parmsDict = None
                if blockName == 'IN':
                    layerInput = None
            yield blockName, layerID, layerInput, ops, parmsDict
            

def decodeConv(ops, opParms, parmsDict):
    """Labels the different parameters for convolutional blocks"""
    defaults = {'kernel' : 3, 'stride' : 1, 'filter' : 1, 'padding' : 1}
    remap = {'k':'kernel', 's':'stride', 'p':'padding', 'f':'filter'}

    parmsDict = {remap[key]:value for (key,value) in opParms.items()}
    parmsDict['batchNorm'] = True if 'BN' in ops else False
    parmsDict['leaky'] = True if 'ReLU' in ops else False
    
    return ChainMap(parmsDict, defaults)

def decodeConcat(ops, opParms, parmsDict):
    """Separates the input from the ops"""
    _, *extraConcats = ops.split(' ')
    parmsDict['inputTensor'] = [parmsDict['inputTensor']] + extraConcats
    return parmsDict