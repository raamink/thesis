#Stdlibs
import unittest
from unittest import TestCase, mock
# import os

#Foreign imports
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

    def test_controller_fileCollection_outputs(self):
        self.assertEqual(self.tester.collectFiles('src/test1/'), ([],[])) # Empty folder
        self.assertEqual(self.tester.collectFiles('src/test2/'), (['parms.json'],['arch.csv'])) # Arch + Parms
        self.assertEqual(self.tester.collectFiles('src/test3/'), ([],['arch.csv', 'arch2.csv'])) # 2 Arch
        self.assertEqual(self.tester.collectFiles('src/test4/'), (['parms.json'],['bla.csv', 'parms2.csv'])) # 2 Parms
        



if __name__ == "__main__":
    unittest.main()