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
from random import randint, choice

#extra libs
from PIL import Image
import numpy as np
import tensorflow as tf
import OpenEXR
import Imath
import array

#local imports
from model import myModel
from EXRparser import exr2depth, exr2flow

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


class dataline:
    def __init__(self, files: str, batchSize: int = 20, batchMode: str = 'random', 
                 inputLabel='flow', outputLabel='masks'):
        if not isinstance(files, (str, Path)):
            raise TypeError('Requires filepath')
        
        if type(files) is str:
            self.rootDir = Path(files)
        else:
            self.rootDir = files
        
        self.labels = ['depth', 'flow', 'masks']
        self.batchSize = batchSize
        self.sequences = [i for i in self.rootDir.glob('clip*') if i.is_dir()]
        trainval = int(len(self.sequences)*0.7)
        self.train = self.sequences[:trainval]
        self.valid = self.sequences[trainval:]

        if batchMode == 'random':
            self.nextBatch = self.nextRandomBatch
        elif batchMode == 'sequential':
            self.nextBatch = self.nextSequentialBatch

        self.dataset = tf.data.Dataset.from_generator(self.nextBatch, output_types=tf.int32)


    def genericBatch(self, sequenceID: str, selectedFrames: list) -> np.array:
        batchLabels = {'frames': [],
                       'depth': [], 
                       'flow': [], 
                       'masks': []}
        
        for frameID in selectedFrames:
            frame, labels = self.collectLabels(sequenceID, frameID)
            for key in labels:
                batchLabels[key].append(labels[key])
        
        return batchFrames, batchLabels


    def nextSequentialBatch(self, nframes: int = None):
        sequence = choice(self.sequences)
        nframes = nframes if nframes else self.batchSize
        batches = self.sequentialBatch(sequence, nframes)
        for batch in batches:
            yield batch

    def sequentialBatch(self, sequenceID: str, nframes: int) -> np.array:
        frameDir = self.rootDir / sequenceID / 'frames'
        numFrames = len(listdir(frameDir))
        frameIDs = [i+1 for i in range(numFrames)]

        while len(frameIDs) >= nframes:
            selFrames, frameIDs = frameIDs[:nframes], frameIDs[nframes:]
            batchFrames, batchLabels = self.genericBatch(sequenceID, selFrames)
            yield batchFrames, batchLabels
            
    def nextRandomBatch(self, nframes: int = None):
        nextSequence = choice(self.sequences)
        nframes = nframes if nframes else self.batchSize
        yield self.randomBatch(nextSequence, nframes)
    
    def randomBatch(self, sequenceID: str, nframes: int = 20) -> np.array:
        frameDir = self.rootDir / sequenceID / 'frames'
        numFrames = len(listdir(frameDir))
        if nframes >= numFrames:
            startFrame, endFrame = 0, numFrames
        else:
            wiggle = randint(0, numFrames - nframes)
            startFrame = 1 + wiggle
            endFrame = startFrame + nframes

        selectedFrames = np.arange(startFrame, endFrame)
        return self.genericBatch(sequenceID, selectedFrames)

    def collectLabels(self, sequenceID: str, imageID: int, dataAug: bool = False) -> list:
        labels = {}
        for label in self.labels:
            imgDir = self.rootDir / sequenceID / label
            if imgDir.is_dir():
                labels[label] = self.singleImage(imgDir, imageID)
        frame = self.singleImage(self.rootDir / sequenceID / 'frames', imageID)

        return frame, labels

    def singleImage(self, imgDir: Path, imageID: int, # label: str, sequenceID: str, imageID: int, 
                    dataAug: bool = False) -> Image:

        imgs = [i for i in imgDir.glob(f'*{imageID:04d}*')]
        lenImgs = len(imgs)
        if lenImgs == 0:
            raise FileNotFoundError(f'No files containing {imageID:04d} in filename at {imgDir}')
        elif lenImgs == 1:
            imgPath = imgs[0]
        else:
            for img in imgs:
                if img.suffix == '.exr':
                    imgPath = img
                    break
        
        if imgPath.suffix == '.exr':
            return self.loadExr(imgPath, dataAug)
        else:
            return self.loadImg(imgPath, dataAug)

    def loadImg(self, imgPath: Path, dataAug: bool = False) -> np.array:
        rawImg = Image.open(imgPath)
        npImg = np.asarray(rawImg)
        return npImg

    def loadExr(self, imgPath: Path, dataAug: bool = False) -> np.array:
        if 'flow' in imgPath.name:
            return self.parseFlowEXR_asCartesian(str(imgPath))
        elif 'depth' in imgPath.name:
            return exr2depth(str(imgPath))

    def parseFlowEXR_asCartesian(self, exrPath: str):
        if issubclass(type(exrPath), Path):
            exrPath = str(exrPath)
        elif type(exrPath) is not str:
            raise TypeError('Filepath needs to be string or Path object.')
        exrFile = OpenEXR.InputFile(exrPath)
        
        # Compute the size
        metadata = exrFile.header()['dataWindow']
        size = (metadata.max.x - metadata.min.x + 1, metadata.max.y - metadata.min.y + 1)
        w, h = size

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        (R,G,B) = [array.array('f', exrFile.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]

        img = np.zeros((h,w,3), np.float64)
        img[:,:,0] = np.array(R).reshape(img.shape[0],-1)
        img[:,:,1] = -np.array(G).reshape(img.shape[0],-1)
        
        return img

    def parseFlowEXR_asPolar(self, exrPath):
        cartesianFlow = self.parseFlowEXR_asCartesian(exrPath)
        
        polarFlow = np.zeros(cartesianFlow.shape, np.uint8)
        polarFlow[...,1] = 255

        mag, ang = cv2.cartToPolar(cartesianFlow[...,0], cartesianFlow[...,1])
        polarFlow[...,0] = ang*180/np.pi/2
        polarFlow[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(polarFlow,cv2.COLOR_HSV2BGR)

        return bgr
