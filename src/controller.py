"""
Main file which controls other aspects of training / running of network.
 - main controller
   - selects network
   - manages training / testing / eval data
 - arg parser for command line interfacing
"""

from os.path import isdir, isfile
from os import listdir
from pathlib import Path

class controller:
    def collectFiles(self, folderName: str):
        if type(folderName) is not str:
            raise ValueError('Wrong input type to collectFiles. Expected string')
        
        cwd = Path.cwd()
        folder = cwd / folderName

        if not folder.exists():
            print(folder, folder.exists)
            raise ValueError('Not a directory')
        
        settings = []
        arches = []
        for entry in folder.glob('./*'):
            if entry.suffix == '.json':
                settings.append(str(entry.relative_to(folder)))
            elif entry.suffix == '.csv':
                arches.append(str(entry.relative_to(folder)))
            else:
                continue

        return settings, arches
    
