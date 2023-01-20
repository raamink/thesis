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
import array

#extra libs
from PIL import Image
import numpy as np
import tensorflow as tf
import OpenEXR
import Imath
from skimage.transform import resize
import cv2

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
    def __init__(self, files: str, batchSize: int = 4, batchMode: str = 'random', 
                 inputs=['flow'], outputs=['masks'], trainTestSplit: float = 0.8):
        if not isinstance(files, (str, Path)):
            raise TypeError('Requires filepath')
        
        if type(files) is str:
            self.rootDir = Path(files)
        else:
            self.rootDir = files
        
        self.labels = ['frames', 'depth', 'flow', 'masks', 'gt_boxes']
        self.batchSize = batchSize
        self.sequences = [i for i in self.rootDir.glob('clip*') if i.is_dir()]
        self.inputs = inputs
        self.outputs = outputs
        self.split = trainTestSplit
        if 'gt_boxes' in outputs:
            self.loadGTLabels()

        self.dataset = self.generateTFDataset(batchSize)

    """
# A rewrite of the dataset generator function to separate the test and train sets. 

    def mapDataset(self):
        trainDS = []
        testDS = []
        for clip in self.sequences:
            frameList = list(i.stem for i in (clip/'frames/'.iterdir()))
            splitClip = int(len(frameList)*self.split)
            trainFrames, testFrames = frameList[:splitClip], frameList[splitClip:]
            trainDS += [(clip, i) for i in trainFrames]
            testDS += [(clip, i) for i in testFrames]
            for i, ID in enumerate(frameList):
                if i<splitClip:
                    trainDS.append((clip, ID))
                else:
                    testDS.append((clip, ID))
        self.trainDS = trainDS
        self.testDS = testDS
            
    def generateDataset(self, batchSize: int = 4, training: bool = True, 
                        inputs=['flow'], outputs=['masks']):
        
        self.mapDataset()
        datasetTuples = self.trainDS if training else self.testDS
        
        dataset = tf.data.Dataset.from_tensor_slices(datasetTuples)
    """
        
    def generateTFDataset(self, batchSize):
        dataset = tf.data.Dataset.from_generator(
            self.datasetGenerator_allClipsTogether,
            output_types=(tf.float32, tf.int16)
        )

        dataset = dataset.repeat()
        dataset = dataset.batch(batchSize)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    
    def datasetGenerator_allClipsTogether(self):
        for sequenceID in self.sequences:
            frameDir = self.rootDir / sequenceID / 'frames'
            clipLength = len(listdir(frameDir))
            frameIDs = [i+1 for i in range(clipLength)]
            
            for element in frameIDs:
                inputs = self.selectLabel(sequenceID, self.inputs, element)
                outputs = self.selectLabel(sequenceID, self.outputs, element)
                yield inputs, outputs

        # self.dataset = tf.keras.preprocessing.timeseries_dataset_from_array(self.nextBatch)
    
    def selectLabel(self, seqID, labels, frameID):
        types = []
        for label in labels:
            imgDir = self.rootDir / seqID / label
            types.append(self.singleImage(imgDir, frameID))
        
        if len(labels) > 1:
            types =np.stack(types)
        else:
            types = np.asarray(types)
        
        types = np.squeeze(types)

        if len(types.shape) == 2:
            types = np.expand_dims(types, -1)

        return types

    def genericBatch(self, sequenceID: str, selectedFrames: list) -> np.array:
        batchFrames = None
        batchLabels = {'frames': [], 
                       'depth': [], 
                       'flow': [], 
                       'masks': [],
                       'gt_labels': []}
        
        for frameID in selectedFrames:
            frame, labels = self.collectLabels(sequenceID, frameID)
            # batchFrames.append(frame)
            for key in labels:
                batchLabels[key].append(labels[key])
        
        inputs = np.squeeze(np.stack([batchLabels[key] for key in self.inputs]))
        outputs = np.squeeze(np.stack([batchLabels[key] for key in self.outputs]))

        return inputs, outputs


    def nextSequentialBatch(self, nframes: int = 1):
        nframes = nframes if nframes else self.batchSize
        # while True:
        #     sequenceID = choice(self.sequences)
        for sequenceID in self.sequences:

            frameDir = self.rootDir / sequenceID / 'frames'
            clipLength = len(listdir(frameDir))
            frameIDs = [i+1 for i in range(clipLength)]

            batches = [frameIDs[i:i+nframes] for i in range(0, len(frameIDs), nframes)]
            # while len(frameIDs) >= nframes:
            #     selFrames, frameIDs = frameIDs[:nframes], frameIDs[nframes:]
            #     yield self.genericBatch(sequenceID, selFrames)
            # else:
            #     continue
            for batch in batches:
                yield self.genericBatch(sequenceID, batch)

            # yield self.sequentialBatch(sequence, nframes)
    # def sequentialBatch(self, sequenceID: str, nframes: int) -> np.array:
    #     frameDir = self.rootDir / sequenceID / 'frames'
    #     numFrames = len(listdir(frameDir))
    #     frameIDs = [i+1 for i in range(numFrames)]

    #     while len(frameIDs) >= nframes:
    #         selFrames, frameIDs = frameIDs[:nframes], frameIDs[nframes:]
    #         yield self.genericBatch(sequenceID, selFrames)
    # def sequentialBatch(self, seqID, selFrames, frameIDs, nframes):
        
    # def _batches_from_IDs(frameIDs, batchSize):
    #         for i in range(0, len(frameIDs), batchSize):
    #             yield frameIDs[]

    def nextRandomBatch(self, nframes: int = None):
        nextSequence = choice(self.sequences)
        nframes = nframes if nframes else self.batchSize
        return self.randomBatch(nextSequence, nframes)
    
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
        # frame = self.singleImage(self.rootDir / sequenceID / 'frames', imageID)
        frame = None

        return frame, labels

    def singleImage(self, imgDir: Path, imageID: int, # label: str, sequenceID: str, imageID: int, 
                    dataAug: bool = False) -> Image:

        if not imgDir.exists():
            if imgDir.name == 'gt_boxes':
                seqID = imgDir.parent.name
                return self.gtlabels[(seqID, imageID)]
            else:
                raise FileNotFoundError((f"Directory {imgDir} does not exist"))

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
            loaded = self.loadExr(imgPath, dataAug)
        else:
            loaded = self.loadImg(imgPath, dataAug)
        
        if len(loaded.shape) == 3:
            newSize = (512, 512, loaded.shape[2])
        else:
            newSize = (512, 512, 1)
        return resize(loaded, newSize)

    def loadImg(self, imgPath: Path, dataAug: bool = False) -> np.array:
        rawImg = Image.open(imgPath).resize((512,512))
        mode = rawImg.mode
        if mode == 'L':
            npImg = np.asarray(rawImg.convert('1'))
        elif mode == 'RGBA':
            npImg = np.asarray(rawImg.convert('RGB'))
        else:
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

        img = np.zeros((h,w,2), np.float64)
        img[:,:,0] = np.array(R).reshape(img.shape[0],-1)
        img[:,:,1] = -np.array(G).reshape(img.shape[0],-1)
        img[np.abs(img)>200] = 0
        
        return img

    def parseFlowEXR_asPolar(self, cartesianFlow):
        # cartesianFlow = self.parseFlowEXR_asCartesian(exrPath)
        cartShape = cartesianFlow.shape
        if cartShape[-1] == 2:
            cartShape = (*cartShape[:-1],3)
        polarFlow = np.zeros(cartShape, np.uint8)
        polarFlow[...,1] = 255

        mag, ang = cv2.cartToPolar(cartesianFlow[...,0], cartesianFlow[...,1])
        polarFlow[...,0] = ang*180/np.pi/2
        polarFlow[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # bgr  = np.zeros(cartShape, np.uint8)
        # if len(cartShape) ==4:
        #     for i, hsv in enumerate(polarFlow[:,...]):
        #         bgr[i,...] = cv2.cvtColor(polarFlow,cv2.COLOR_HSV2BGR)
        # else:
        #     bgr = cv2.cvtColor(polarFlow, cv2.COLOR_HSV2BGR)

        # return bgr
        return polarFlow

    def loadGTLabels(self) -> np.array:
        self.gtlabels = {}

        #find ground truth labels file
        boxfile = self.rootDir/'gt_boxes.txt'
        if not boxfile.exists():
            raise FileNotFoundError()
        
        #load ground truth boxes
        with boxfile.open("r") as data:
            lines = data.readlines()
        
        for line in lines:
            image, *boxStrings = line.split(" ")

            #extract clipID and frameNum for dict key
            image = '/Data' / Path(image)
            im = Image.open(image)
            origWidth,origHeight = im.size
            im.close()
            clip = image.parent.parent.name
            frameID = int(image.name.split('.')[0])

            #extract and parse boxes into tuples
            boxes = []
            for box in boxStrings:
                x1,y1,x2,y2,c = np.fromstring(box, sep=',', dtype=np.int)
                x3 = (x1 * 512) // origWidth
                x4 = (x2 * 512) // origWidth
                y3 = (y1 * 512) // origHeight
                y4 = (y2 * 512) // origHeight
                boxes.append((x3,y3,x4,y4,c))
            
            #add to dictionary
            self.gtlabels[(clip, frameID)] = boxes
    



# def testFor10k(path):
#     exr = OpenEXR.InputFile(path)
#     Float = Imath.PixelType(Imath.PixelType.FLOAT)
#     rgb = [array.array('f', exr.channel(C, Float)).tolist() for C in 'RGB']
#     img = np.zeros((1080,1920,3), np.float16)
#     for i, C in enumerate(rgb):
#         img[:,:,i] = np.array(C).reshape(img.shape[0], -1)
#     return True if np.max(img) == 10000.0 else False
# while num<202 and Failing:
#     path = f'/Data/dataset-OrigFiles/clip2v2/flow/flow0{num:03}.exr'
#     Failing = testFor10k(path)
#     print(f'Is {num} poisoned?\t: {Failing}')
#     num += 1

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("\n\n1\n\n")

    dataset = Path('/Data/dataset-OrigFiles/')
    clips = dataset.glob("clip8")
    
    batchSize = 6
    dataloader = dataline(dataset, batchSize, batchMode='sequential', inputs=['frames'], outputs=['masks'])
    
    print("\n\ndataloader created\n\n")

    for clip in clips:
        print(clip)
        dataloader.sequences = [clip.name]
        batches = dataloader.nextSequentialBatch(6)
        # dataloader.datasetGenerator_allClipsTogether()
        # batches = dataloader.dataset.as_numpy_iterator()

        frames, boxes = next(batches)
        counter = 0
        # print('ping')
        # for flowArrays, maskArrays, in batches:
        #     counter += 1
        #     print(counter)
        #     # if counter == 348:
        #     #     break
        #     plt.figure()
        #     plt.suptitle(f'Batch #{counter}')
        #     for i in range(batchSize):
        #         plt.subplot(batchSize, 3, 3*i+1)
        #         plt.imshow(flowArrays[i,..., 0])
        #         plt.title(f'flow_x {i+1}')
        #         plt.colorbar()

        #         plt.subplot(batchSize, 3, 3*i+2)
        #         plt.imshow(flowArrays[i,..., 1])
        #         plt.title(f'flow_y {i+1}')
        #         plt.colorbar()

        #         plt.subplot(batchSize, 3, 3*i+3)
        #         plt.imshow(maskArrays[i,...])
        #         plt.title(f'mask {i+1}')
        #         plt.colorbar()
        #     plt.show()
        for frameArrays, maskArrays in batches:
            counter += 1
            print(counter, '\t', frameArrays.shape, maskArrays.shape)
            if frameArrays.shape[-1] == 1:
                print('\t\t\t\t', clip, counter)
            break
            # plt.figure()
            # plt.suptitle(f'Batch #{counter} of {clip.name}')
            # for i in range(batchSize):
            #     plt.subplot(batchSize, 2, 2*i+1)
            #     plt.imshow(frameArrays[i,...])
            #     plt.title(f'RGB {i+1}, {frameArrays[i,...].shape}')

            #     plt.subplot(batchSize, 2, 2*i+2)
            #     plt.imshow(maskArrays[i,...])
            #     plt.title(f'mask {i+1}, {maskArrays[i,...].shape}')
            # plt.show()
            # break
        

"""
clip6 seems to not have flow data?


"""