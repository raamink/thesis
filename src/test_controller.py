#Stdlibs
import unittest
from unittest import TestCase, mock
# import os

#Foreign imports
# import tensorflow as tf
# import tensorflow.keras as keras
# import numpy as np

#Local imports
import controller

class testController(TestCase):
    def setUp(self):
        self.tester = controller.controller

    def test_controller_fileCollection_inputs(self):
        self.assertRaises(ValueError, self.tester.collectFiles, 1)       # Int
        self.assertRaises(ValueError, self.tester.collectFiles, 1.0)     # Float
        self.assertRaises(ValueError, self.tester.collectFiles, [1])     # List
        self.assertRaises(ValueError, self.tester.collectFiles, True)    # Bool
        self.assertRaises(ValueError, self.tester.collectFiles, {1:1})   # Dict
        self.assertRaises(ValueError, self.tester.collectFiles, (1,1))   # Tuple
        
        self.assertRaises(ValueError, self.tester.collectFiles, 'balkjdaf') # Folder doens't exist
        
    def test_controller_fileCollection_outputs(self):
        self.assertEqual(self.tester.collectFiles('src/test1/'), ([],[])) # Empty folder
        self.assertEqual(self.tester.collectFiles('src/test2/'), (['parms.json'],['arch.csv'])) # Arch + Parms
        self.assertEqual(self.tester.collectFiles('src/test3/'), ([],['arch.csv', 'arch2.csv'])) # 2 Arch
        self.assertEqual(self.tester.collectFiles('src/test4/'), (['parms.json'],['bla.csv', 'parms2.csv'])) # 2 Parms
        

        

if __name__ == "__main__":
    unittest.main()