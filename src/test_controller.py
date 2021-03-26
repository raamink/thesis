#Stdlibs
import unittest
from unittest import TestCase, mock
# import os

#Foreign imports
from PIL import Image
import numpy as np
from pathlib import Path
# import tensorflow as tf
# import tensorflow.keras as keras
# import numpy as np

#Local imports
import controller
import model

class testController(TestCase):
    def setUp(self):
        self.tester = controller.controller()

    def test_collectFiles_inputs(self):
        self.assertRaises(ValueError, self.tester.collectFiles, 1)       # Int
        self.assertRaises(ValueError, self.tester.collectFiles, 1.0)     # Float
        self.assertRaises(ValueError, self.tester.collectFiles, [1])     # List
        self.assertRaises(ValueError, self.tester.collectFiles, True)    # Bool
        self.assertRaises(ValueError, self.tester.collectFiles, {1:1})   # Dict
        self.assertRaises(ValueError, self.tester.collectFiles, (1,1))   # Tuple
        
        self.assertRaises(ValueError, self.tester.collectFiles, 'balkjdaf') # Folder doens't exist
        
    def test_collectFiles_outputs(self):
        self.assertEqual(self.tester.collectFiles('tests/test1/'), 
                         ([],[])) # Empty folder
        self.assertEqual(self.tester.collectFiles('tests/test2/'), 
                         ([Path('tests/test2/parms.json')],[Path('tests/test2/arch.csv')])) # Arch + Parms
        self.assertEqual(self.tester.collectFiles('tests/test3/'), 
                         ([],[Path('tests/test3/arch.csv'), Path('tests/test3/arch2.csv')])) # 2 Arch
        self.assertEqual(self.tester.collectFiles('tests/test4/'), 
                         ([Path('tests/test4/parms.json')],[Path('tests/test4/bla.csv'), Path('tests/test4/parms2.csv')])) # 2 Parms

    def test_createsMyModelObject(self):
        files = ('tests/testArch.csv','tests/testCompileParms.json')
        output = self.tester.network(files)
        self.assertIsInstance(output, model.myModel)
        self.assertTrue(output.model._is_compiled)

    def test_manageSequenceOfMyModels(self):
        files = (['a.json'],
                 ['b.csv', 'c.csv'])

        with mock.patch('controller.controller.network', lambda self,x: x):
            output = list(self.tester.nextNetwork(files))
        
        self.assertEqual(output, [('b.csv', 'a.json'), ('c.csv', 'a.json')])

class testDataLine(TestCase):
    def setUp(self):
        testDir = '/Data/Thesis/Python/tests/'
        self.tester = controller.dataline(testDir)

    def test_returns_singleImage(self):
        imgDir = Path('/Data/Thesis/Python/tests/clipTest/frames')
        expected = Path('/Data/Thesis/Python/tests/clipTest/frames/0001.png')
        with mock.patch('controller.dataline.loadImg', lambda x,y,z: y):
            with mock.patch('controller.dataline.loadExr', lambda x,y,z: y):
                output = self.tester.singleImage(imgDir, 1)
        self.assertEqual(output, expected)

        imgDir = Path('/Data/Thesis/Python/tests/clipTest/flow')
        expected = Path('/Data/Thesis/Python/tests/clipTest/flow/flow0001.exr')
        with mock.patch('controller.dataline.loadImg', lambda x,y,z: y):
            with mock.patch('controller.dataline.loadExr', lambda x,y,z: y):
                output = self.tester.singleImage(imgDir, 1)
        self.assertEqual(output, expected)



    def test_returns_allLabels(self):
        expected = (None,
                   {'frames': Path('/Data/Thesis/Python/tests/clipTest/frames/'),
                    'depth': Path('/Data/Thesis/Python/tests/clipTest/depth/'),
                    'flow': Path('/Data/Thesis/Python/tests/clipTest/flow/'),
                    'masks': Path('/Data/Thesis/Python/tests/clipTest/masks/')})
        with mock.patch('controller.dataline.singleImage', lambda s, p, i: p):
            output = self.tester.collectLabels('clipTest', 1)
        self.assertEqual(output, expected)


    def test_returns_randomBatch(self):
        inputs, outputs = self.tester.randomBatch('clipTest', nframes=3)
        self.assertTrue(isinstance(inputs, np.ndarray))
        self.assertTrue(isinstance(outputs, np.ndarray))
        self.assertEqual((3, 1080, 1920, 3), inputs.shape)
        self.assertEqual((3, 1080, 1920), outputs.shape)
    
    def test_randomBatchIterator(self):
        def counter():
            i = 0
            while True:
                i += 1
                yield i, i, i, i
        
        with mock.patch('controller.dataline.randomBatch', counter):
            batch1 = self.tester.nextBatch()
            batch2 = self.tester.nextBatch()

        self.assertNotEqual(batch1, batch2)

    def test_sequenceConsumption(self):
        def counter(_, __, selFrames):
            nframes = len(selFrames)
            i = [1,2,3,4,5,6,7,8,9]
            returns = []
            for frame in selFrames:
                returns.append(i[frame-1])
            return returns, {0:[0]}
        
        with mock.patch('controller.dataline.genericBatch', counter):
            batches = list(self.tester.sequentialBatch('clipTest', 3))
            output = [batch[0] for batch in batches]
        expected = [[1,2,3],[4,5,6]]
        
        self.assertEqual(output, expected)

    def test_sequentialBatchIterator(self):
        def mockCollectLabels(s, sID, fID):
            labels = {'frames': fID,
                      'depth' : fID,
                      'flow' : fID,
                      'masks' : fID}
            return None, labels
        
        with mock.patch('controller.dataline.collectLabels', mockCollectLabels):
            with mock.patch('controller.listdir', lambda x: [i for i in range(10)]):
                batcher = self.tester.sequentialBatch('sid', 3)
                # print('PRINTY: ', batcher)
                batch1 = next(batcher)
                batch2 = next(batcher)
                batch3 = next(batcher)
                # print('PRINTY2: ', batch1)
                self.assertRaises(StopIteration, next, batcher)
        
        self.assertTrue(isinstance(batch1, tuple))
        self.assertEqual(len(batch1), 2)

        expected1 = (np.array([1,2,3]), np.array([1,2,3]))
        expected2 = (np.array([4,5,6]), np.array([4,5,6]))
        expected3 = (np.array([7,8,9]), np.array([7,8,9]))

        for i in range(2):
            self.assertTrue(np.array_equal(batch1[i], expected1[i]))
            self.assertTrue(np.array_equal(batch2[i], expected2[i]))
            self.assertTrue(np.array_equal(batch3[i], expected3[i]))
        
        with mock.patch('controller.dataline.collectLabels', mockCollectLabels):
            with mock.patch('controller.listdir', lambda x: [i for i in range(10)]):
                batcher = self.tester.sequentialBatch('sid', 4)
                batches = [batch for batch in batcher]
                self.assertEqual(len(batches), 2)

    def test_parseFlow(self):
        shortCallable = self.tester.parseFlowEXR_asCartesian

        self.assertRaises(TypeError, shortCallable, 1)
        self.assertRaises(TypeError, shortCallable, [1])
        self.assertRaises(TypeError, shortCallable, 1.0)
        self.assertRaises(TypeError, shortCallable, {1:1})
        self.assertRaises(TypeError, shortCallable, )

        testFile = "tests/clipTest/flow/flow0001.exr"
        expectedShape = (1080, 1920, 3)

        output = shortCallable(testFile)

        self.assertTrue(isinstance(output, np.ndarray))
        self.assertEqual(output.shape, expectedShape)

    def test_parseDepth(self):
        shortCallable = self.tester.parseDepthEXR

        self.assertRaises(TypeError, shortCallable, 1)
        self.assertRaises(TypeError, shortCallable, [1])
        self.assertRaises(TypeError, shortCallable, 1.0)
        self.assertRaises(TypeError, shortCallable, {1:1})
        self.assertRaises(TypeError, shortCallable, )

        testFile = "tests/clipTest/depth/depth0001.exr"
        expectedShape = (1080, 1920)

        output = shortCallable(testFile)

        self.assertTrue(isinstance(output, np.ndarray))
        self.assertEqual(output.shape, expectedShape)



if __name__ == "__main__":
    unittest.main()