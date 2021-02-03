"""Attempted TTD module for the pipeline."""

# Stdlibs
import unittest
from unittest import TestCase
from unittest import mock
from collections import ChainMap

# Foreign imports
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

#Local imports
import model
model.printInstantiation = False
model.printBuildInputs = False

class testMyModel(TestCase):

    def setUp(self):
        self.testModel = model.myModel()

    def test_model_initialises(self):
        self.assertEqual(dict(), self.testModel.architecture)

    def test_model_collectBuilders(self):
        self.assertFalse(hasattr(self.testModel, 'builders'))

        self.testModel.collectBuilders()

        self.assertIsNotNone(self.testModel.builders.layerTypes, 
                             msg='Failed to collect any builders')
        self.assertEqual(len(self.testModel.builders.layerTypes), 
                         len(model.layerFactory.__subclasses__()), 
                         msg='Failed to collect all builers')
    
    def test_model_buildArchitecture(self):
        self.testModel.collectBuilders()
        self.testModel.buildArchitecture('architectures/testArch.csv')

        self.assertNotEqual(self, dict(), self.testModel.architecture)
        self.assertIn('L0', self.testModel.architecture)
        self.assertIn('LN', self.testModel.architecture)
        self.assertTrue(hasattr(self.testModel, 'model'))
        self.assertIsInstance(self.testModel.model, tf.keras.Model)
        self.assertTrue(hasattr(self.testModel.model, 'compile'))

    def test_model_collectCompileParms(self):
        self.assertTrue(hasattr(self.testModel, 'collectCompileParms'))
        testFunction = self.testModel.collectCompileParms
        self.assertRaises(NotImplementedError, testFunction, 'bla')
        self.assertRaises(FileNotFoundError, testFunction, 'bla.json')
        with mock.patch('model.json.load') as mock_json_load:
            mock_json_load.return_value = {'optimizer': 'adam'}
            with mock.patch('model.open', mock.mock_open(read_data='')):
                testFunction('architectures/testCompileParms.json')
        expected = {'optimizer': 'adam', 
                    'loss': 'sparse_categorical_crossentropy', 
                    'metrics': ['sparse_categorical_accuracy']}
        self.assertTrue(isinstance(self.testModel.compileParms, dict))
        self.assertEqual(self.testModel.compileParms, expected)
        del self.testModel.compileParms

        with mock.patch('model.json.load') as mock_json_load:
            mock_json_load.return_value = {'optimizer': 'sgd', 'learn_rate': 1e-5}
            with mock.patch('model.open', mock.mock_open(read_data='')):
                testFunction('bla.json')
        expected = {'optimizer': 'sgd', 
                    'loss': 'sparse_categorical_crossentropy', 
                    'metrics': ['sparse_categorical_accuracy'],
                    'learn_rate': 1e-5}
        self.assertEqual(self.testModel.compileParms, expected)

    def test_model_compileModel(self):
        self.assertTrue(hasattr(self.testModel, 'compileModel'))
        self.assertRaises(ValueError, self.testModel.compileModel)
        self.testModel.model = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
        self.assertRaises(KeyError, self.testModel.compileModel)
        self.testModel.compileParms = {'optimizer': 'rmsprop', 
                    'loss': 'sparse_categorical_crossentropy', 
                    'metrics': ['sparse_categorical_accuracy']}
        self.assertFalse(self.testModel.model._is_compiled)
        self.testModel.compileModel()
        self.assertTrue(self.testModel.model._is_compiled)

    @mock.patch('model.myModel.compileModel')
    @mock.patch('model.myModel.collectCompileParms')
    @mock.patch('model.myModel.buildArchitecture')
    def test_model_compiles(self, mock_arch, mock_parms, mock_compile):
        self.assertFalse(mock_compile.called)
        test = model.myModel('bla.csv', 'bla.json')
        self.assertTrue(mock_compile.called)


class testFileIO(TestCase):
    def test_buildLineIterator(self):
        expectedOutput = [('IN', 'L0', '-', '-', ChainMap({'shape': (None, None, 3)},{})),
                          ('Conv1', 'L1', 'L0', 'Conv+BN+ReLU', ChainMap({'filter':16, 'kernel':5, 'stride':1, 'leaky':True, 'batchNorm': True},{'padding': 'valid'})),
                          ('resnet1', 'L1', 'L0', 'Conv+ReLU', ChainMap({'filter':16, 'kernel':1, 'leaky':True, 'batchNorm': False},{'padding': 'valid', 'stride':1})),
                          ('resnet1', 'L2', 'L1', 'Conv+ReLU', ChainMap({'filter':16, 'kernel':3, 'leaky':True, 'batchNorm': False},{'padding': 'valid', 'stride':1})),
                          ('resnet1', 'L3', 'L2', 'Conv+ReLU', ChainMap({'filter':3, 'kernel':1, 'leaky':True, 'batchNorm': False},{'padding': 'valid', 'stride':1})),
                          ('Concat', 'L4', 'L3', 'Concat L1', {'concatWith': ['L1']}),
                          ('maxpool', 'L5', 'L4', 'MaxPooling', ChainMap({},{'pool_size': (2,2), 'strides': None})),
                          ('OUT', 'LN', 'L4', '-' , ChainMap({},{'shape': (None, None, 3)}))]

        output = list(model.buildLineIterator('architectures/testArch.csv'))
        self.assertEqual(len(expectedOutput), len(output))
        self.assertEqual(expectedOutput, output)

    def test_decode(self):
        blockName = 'Concat'
        ops = 'Conv+ReLU'
        self.assertEqual(model.decodeConcat, model.decode(blockName, ops))

        blockName = 'maxpool'
        self.assertEqual(model.decodeMaxPool, model.decode(blockName, ops))

        blockName = 'resnet1'
        self.assertEqual(model.decodeConv, model.decode(blockName, ops))

        blockName = 'Spam'
        self.assertRaises(ValueError, model.decode, blockName, ops)

    def test_decodIn(self):
        ops = '-'

        opParms = {}
        expected = {'shape': (None, None, 3)}
        self.assertEqual(expected, model.decodeIn(ops, opParms, {}))

        opParms = {'s':'(52,)'}
        expected = {'shape': (52,)}
        self.assertEqual(expected, model.decodeIn(ops, opParms, {}))

        opParms = {'bla':'bla'}
        expected = {'shape': (None, None, 3)}
        self.assertEqual(expected, model.decodeIn(ops, opParms, {}))

    def test_decodeConv(self):
        # 3 cases for decodeConv
        ops = 'Conv+BN+ReLU'
        opParms = {'f':'16', 'k':'5', 's':'1', 'p':'"same"'}
        expected = {'batchNorm':True, 'leaky':True, 'filter':16, 'kernel': 5, 
                    'stride':1, 'padding': 'same'}
        self.assertEqual(expected, model.decodeConv(ops, opParms, {}))

        ops = 'Conv'
        opParms = {}
        expected = {'batchNorm':False, 'leaky':False, 'kernel': 3, 'filter':1,
                    'stride':1, 'padding': 'valid'}
        self.assertEqual(expected, model.decodeConv(ops, opParms, {}))

        opParms = {'d':'16'}
        self.assertEqual(expected, model.decodeConv(ops, opParms, {}))

    def test_decodeConcat(self):
        # Expected usecase
        parmsDict = {}
        opParms = {}

        ops = 'Concat L0'
        expected = {'concatWith': ['L0']}
        self.assertEqual(expected, model.decodeConcat(ops, opParms, parmsDict))

        ops = 'Concat L0 L2'
        expected = {'concatWith': ['L0', 'L2']}
        self.assertEqual(expected, model.decodeConcat(ops, opParms, parmsDict))

        # Expected mistake
        ops = 'Concat'
        self.assertRaises(ValueError, model.decodeConcat, ops, opParms, parmsDict)

    def test_decodeMaxPool(self):
        ops = "MaxPooling"

        parmsDict = {}
        opParms = {}
        expected = {'pool_size':(2,2), 'strides':None}
        self.assertEqual(expected, model.decodeMaxPool(ops, opParms, parmsDict))

        opParms = {'ps=1', 's=1'}
        expected = {'pool_size':(1,1), 'strides':(1,1)}
        self.assertEqual(expected, model.decodeMaxPool(ops, opParms, parmsDict))

        opParms = {'ps=(1,1)', 's=(1,2)'}
        expected = {'pool_size':(1,1), 'strides':(1,2)}
        self.assertEqual(expected, model.decodeMaxPool(ops, opParms, parmsDict))

        opParms = {'random="val"'}
        parmsDict = model.decodeMaxPool(ops, opParms, parmsDict)
        self.assertRaises(KeyError, lambda x: parmsDict[x], 'random')

class testFactories(tf.test.TestCase):
    def test_inputFactory(self):
        factory = model.inputFactory()
        arch = {}
        parms = {'shape': (None,None,3)}
        blockParms = [['L0', ['-', '-', parms]]]
        
        self.assertIsNotNone(factory.buildLayer(['-', '-', parms]))
        self.assertEqual(type(tf.keras.layers.Input(shape=(None, None, 3))), type(factory.buildLayer(['-', '-', parms])))

        factory.buildBlock(blockParms=blockParms, architecture=arch)
        self.assertEqual(len(arch), 1)

    def test_convFactory(self):
        factory = model.convFactory()
        arch = {'L0': tf.keras.layers.Input(shape=(None, None, 3))}
        blockParms = [['L1', ['L0', 'conv+BN+ReLU', {'filter': 16, 'kernel': 5, 'stride': 1, 'padding': 'same', 
                        'batchNorm': True, 'leaky': True}]]]

        factory.buildBlock(blockParms, arch)
        self.assertNotEqual({'L0': tf.keras.layers.Input(shape=(None, None, 3))}, arch)
        self.assertIsNotNone(arch['L1'])

    def test_resnetFactory(self):
        factory = model.resnetFactory()
        arch = {'L0': tf.keras.layers.Input(shape=(None, None, 3))}
        arch2 = {'L0': tf.keras.layers.Input(shape=(None, None, 3))}
        blockParms = [
                ['L1', ['L0', 'conv+BN+ReLU', {'filter': 16, 'kernel': 5, 'stride': 1, 'padding': 'same', 
                        'batchNorm': True, 'leaky': True}]],
                ['L2', ['L1', 'conv+BN+ReLU', {'filter': 16, 'kernel': 5, 'stride': 1, 'padding': 'valid', 
                        'batchNorm': True, 'leaky': True}]],
                ['L3', ['L2', 'conv+BN+ReLU', {'filter': 3, 'kernel': 3, 'stride': 1, 'padding': 'same', 
                        'batchNorm': True, 'leaky': True}]]]
        
        factory.buildBlock(blockParms, arch)
        
        self.assertNotEqual(arch, arch2)
        self.assertEqual(len(arch), 4)

    def test_concatFactory_buildLayer(self):
        factory = model.concatFactory()
        
        input_x = np.arange(9).reshape((3,1,3))
        expectedOutput = np.concatenate([input_x, input_x], axis=-1)
        
        arch = {'L0': tf.constant(np.arange(9).reshape((3,1,3))),
                'L1': tf.constant(np.arange(9).reshape((3,1,3)))}

        _, layerParms = ['L2', ['L1', 'Concat L0', {'concatWith': 'L0'}]]
        
        inputTensor = arch['L1']

        newlayer = factory.buildLayer(layerParms, inputTensor, arch)

        self.assertAllEqual(expectedOutput, newlayer)
        self.assertShapeEqual(expectedOutput, newlayer)

        layerParms = ['L1', 'Concat L0', {'concatWith': ['L0']}]
        newlayer = factory.buildLayer(layerParms, inputTensor, arch)

        self.assertAllEqual(expectedOutput, newlayer)
        self.assertShapeEqual(expectedOutput, newlayer)


        arch = {'L0': tf.constant(np.arange(9).reshape((3,1,3))),
                'L1': tf.constant(np.arange(9).reshape((3,1,3))),
                'L2': tf.constant(np.arange(9).reshape((3,1,3)))}

        expectedOutput = np.concatenate([input_x, input_x, input_x], axis=-1)
        layerParms = ['L2', 'Concat L0', {'concatWith': ['L0', 'L1']}]
        inputTensor = arch['L2']

        newlayer = factory.buildLayer(layerParms, inputTensor, arch)

        self.assertAllEqual(expectedOutput, newlayer)
        self.assertShapeEqual(expectedOutput, newlayer)

    def test_concatFactory_buildBlock(self):
        factory = model.concatFactory()

        arch = {'L0': tf.constant(np.arange(9).reshape((3,1,3))),
                'L1': tf.constant(np.arange(9).reshape((3,1,3)))}
        
        oldSize = len(arch)
        blockParms = [['L2', ['L1', 'Concat L0', {'concatWith': 'L0'}]]]

        factory.buildBlock(blockParms, arch)

        self.assertGreater(len(arch), oldSize)

    def test_outputFactory_buildLayer(self):
        factory = model.outputFactory()

        _, layerParms = ['LN', ['L1', '-', {}]]

        arch = {'L1': tf.constant(np.arange(9).reshape((3,1,3)))}

        expected = tf.constant(np.arange(9).reshape((3,1,3)))

        newlayer = factory.buildLayer(layerParms, arch)

        self.assertAllEqual(expected, newlayer)

    def test_outputFactory_buildBlock(self):
        factory = model.outputFactory()

        arch = {'L0': tf.constant(np.arange(9).reshape((3,1,3)))}
        
        oldSize = len(arch)
        blockParms = [['LN', ['L0', '-', {}]]]

        factory.buildBlock(blockParms, arch)

        self.assertGreater(len(arch), oldSize)

        arch = {'L0': tf.constant(np.arange(9).reshape((3,1,3))),
                'L1': tf.constant(np.arange(3).reshape((1,1,3)))}

        blockParms = [['LN', ['L0', '-', {}]]]

    def test_maxpoolFactory_buildLayer(self):
        factory = model.maxpoolFactory()

        x1 = np.arange(16).reshape((1,4,4,1))
        y1 = np.array([[[[5],[7]],[[13],[15]]]])

        X1 = tf.constant(x1)
        layerParms = {'pool_size': (2,2), 'strides': None}

        Y1 = factory.buildLayer(layerParms, X1)

        self.assertIsNotNone(Y1)
        self.assertAllEqual(y1, Y1)

        y2 = np.arange(16).reshape((4,4))[1:,1:].reshape((1,3,3,1))
        # layerParms = ['L1', 'MaxPooling', {'pool_size': (2,2), 'strides': (1,1)}]
        layerParms = {'pool_size': (2,2), 'strides': (1,1)}
        
        Y2 = factory.buildLayer(layerParms, x1)

        self.assertAllEqual(y2, Y2)

        y3 = np.arange(16).reshape((4,4))[1::2,1:].reshape((1,2,3,1))
        # layerParms = ['L1', 'MaxPooling', {'pool_size': (2,2), 'strides': (2,1)}]
        layerParms = {'pool_size': (2,2), 'strides': (2,1)}
        
        Y3 = factory.buildLayer(layerParms, x1)

        self.assertAllEqual(y3, Y3)

    def test_maxPoolFactory_buildBlock(self):
        factory = model.maxpoolFactory()

        arch = {'L1': tf.constant(np.ones((1,4,4,1)))}
        blockParms = [['L2', ['L1', 'maxpool', {'pool_size': (2,2), 'strides': None}]]]

        factory.buildBlock(blockParms, arch)

        self.assertIn('L2', arch)
        self.assertAllEqual(np.ones((1,2,2,1)), arch['L2'])

    @mock.patch('model.maxpoolFactory.buildBlock')
    @mock.patch('model.outputFactory.buildBlock')
    @mock.patch('model.concatFactory.buildBlock')
    @mock.patch('model.resnetFactory.buildBlock')
    @mock.patch('model.convFactory.buildBlock')
    @mock.patch('model.inputFactory.buildBlock')
    def test_genericFactory_buildBlock(self, mock_input, mock_conv, mock_resnet, 
                mock_concat, mock_output, mock_maxpool):
        factory = model.layerFactory()

        for cls in model.layerFactory.__subclasses__():
            cls()
        
        self.assertFalse(mock_input.called)
        factory.buildBlock('IN', [], {})
        self.assertTrue(mock_input.called, "`inputFactory.buildBlock` not called with `blockType` 'IN'")
        
        self.assertFalse(mock_conv.called)
        factory.buildBlock('Conv',[], {})
        self.assertTrue(mock_conv.called, "`convFactory.buildblock` not called with `blockType` 'Conv'")
        
        self.assertFalse(mock_resnet.called)
        factory.buildBlock('resnet',[], {})
        self.assertTrue(mock_resnet.called, "`resnetFactory.buildblock` not called with `blockType` 'resnet'")
        
        self.assertFalse(mock_concat.called)
        factory.buildBlock('Concat', [], {})
        self.assertTrue(mock_concat.called)

        self.assertFalse(mock_output.called)
        factory.buildBlock('OUT', [], {})
        self.assertTrue(mock_output.called)

        self.assertFalse(mock_maxpool.called)
        factory.buildBlock('maxpool', [], {})
        self.assertTrue(mock_maxpool.called)

        self.assertRaises(KeyError, factory.buildBlock,'nonexistentBlockType',[], {})
        

if __name__ == "__main__":
    unittest.main()

