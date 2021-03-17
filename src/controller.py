"""
Main file which controls other aspects of training / running of network.
 - main controller
   - selects network
   - manages training / testing / eval data
 - arg parser for command line interfacing
"""

#stdlibs
from os.path import isdir, isfile
from os import listdir
from pathlib import Path

#local imports
from model import myModel

class controller:
    def __init__(self, archFolder: str = None):
        if archFolder:
            self.archFolder = Path(archFolder)
            self.archFiles = self.collectFiles(archFolder)

    def collectFiles(self, folderName: str):
        if type(folderName) is not str:
            raise ValueError('Wrong input type to collectFiles. Expected string')
        
        # cwd = Path.cwd()
        folder = Path(folderName)

        if not folder.exists():
            print(folder, folder.exists)
            raise ValueError('Not a directory')
        
        settings = []
        arches = []
        for entry in folder.glob('./*'):
            if entry.suffix == '.json':
                # settings.append(str(entry.relative_to(folder)))
                settings.append(entry)
            elif entry.suffix == '.csv':
                # arches.append(str(entry.relative_to(folder)))
                arches.append(entry)
            else:
                continue

        return settings, arches

    def network(self, files: tuple) -> myModel:
        archFile, compFile = files
        return myModel(architectureFile=archFile, compileFile=compFile)

    def nextNetwork(self, files: tuple) -> myModel:
        parms, arches = files
        for parm in parms:
            for arch in arches:
                yield self.network((arch, parm))

