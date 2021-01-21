"""Attempted TTD module for the pipeline."""

# Stdlibs
import unittest
from unittest import TestCase
from unittest import mock
from collections import ChainMap

# Foreign imports
import tensorflow as tf

#Local imports
import model
model.printInstantiation = False
model.printBuildInputs = True

class testMyModel(TestCase):

    def test_modelInits(self):
        empty = model.myModel()
        self.assertEqual(dict(), empty.architecture)
    
    # def test_modelMakesArchitecture(self):
    #     testArch = model.myModel('architectures/testArch.csv')
    #     self.assertNotEqual(self, dict(), testArch.architecture)

    def test_collectBuilders(self):
        empty = model.myModel()
        self.assertIsNotNone(empty.builders.layerTypes)
        print(empty.builders.layerTypes)
    
    def test_decodIn(self):
        # 3 cases for decodeIn
        readData = 'IN, L0, -, -\nIN, L0, -, -, s=(52,)\nIN, L0\n'
        with mock.patch('model.open', mock.mock_open(read_data=readData)):
            line = model.buildLineIterator('spam')

            expected = ('IN', 'L0', '-', '-', ChainMap({}, {'shape': (None, None, 3)}))
            self.assertEqual(expected, next(line))
        
            expected = ('IN', 'L0', '-', '-', ChainMap({'shape': (52,)}, {'shape': (None, None, 3)}))
            self.assertEqual(expected, next(line))
        
            with self.assertRaises(ValueError):
                next(line)

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

        ops = 'Concat L0'
        expected = ('Concat', {'concatWith': ['L0']})
        self.assertEqual(expected, model.decodeConcat(ops, parmsDict))

        ops = 'Concat L0 L2'
        expected = ('Concat', {'concatWith': ['L0', 'L2']})
        self.assertEqual(expected, model.decodeConcat(ops, parmsDict))

        # Expected mistake
        ops = 'Concat'
        self.assertRaises(ValueError, model.decodeConcat, ops, parmsDict)

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


if __name__ == "__main__":
    unittest.main()

