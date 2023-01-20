from controller import controller, dataline

if __name__ == "__main__":
    import tensorflow as tf
    
    from datetime import datetime
    from pathlib import Path
    import os
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    rootDir = Path('/Data')
    archFiles = str(Path.cwd() / 'new')
    # archFiles = str(rootDir / )
    logDir = rootDir / 'logs'
    dataDir = rootDir / 'dataset-OrigFiles'
    netName = 'UNet-RGB-bcelf-'

    control = controller(archFiles)

    networks = control.nextNetwork(control.archFiles)

    trainData = dataline(dataDir, batchMode='sequential', inputs=['frames'], outputs=['masks'])
    sequences = trainData.sequences
    sequences.remove(dataDir/'clip13')
    trainData.sequences, unusedSeq= sequences[:-6], sequences[-6:]
    unused = iter(unusedSeq)
    epochsteps = 165

    # valData = dataline('/Data/dataset2/')
    # valData.sequences = ['clip2']

    batchSize=2


    for network in networks:
        print(network)
        print(network.model.summary())
        print(network.files)

        identifier =  datetime.now().strftime("%Y%m%d-%H%M%S")
        logs = str(logDir / identifier)
        model = str( rootDir / 'models' / (netName + identifier))
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = 2)

        for i in range(5):
            if i<3:
                trainData.sequences.append(unused)
                trainData.sequences.append(unused)
                epochsteps += 54

            print(f'\t\tCounter: {i+1} of 5')
            network.model.fit(trainData.dataset, epochs=5*(i+1), steps_per_epoch=epochsteps,
                              verbose=1, batch_size=batchSize, callbacks = [tboard_callback],
                              initial_epoch=5*i)
            network.model.save(model)
