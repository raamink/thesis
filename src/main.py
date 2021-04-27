from controller import controller, dataline
import tensorflow as tf
from datetime import datetime
from pathlib import Path


if __name__ == "__main__":
    rootDir = Path('/Data')
    archFiles = str(rootDir / 'models/new')
    logDir = rootDir / 'logs'
    dataDir = rootDir / 'dataset_flowMask'

    control = controller(archFiles)

    networks = control.nextNetwork(control.archFiles)

    trainData = dataline(dataDir, batchMode='sequential')

    batchSize=2


    for network in networks:
        print(network)
        print(network.model.summary())

        identifier =  datetime.now().strftime("%Y%m%d-%H%M%S")
        logs = str(logDir / identifier)
        model = str( rootDir / 'models' / identifier)
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = 2)

        for i in range(5):
            print(f'\t\tCounter: {i+1} of 5')
            network.model.fit(trainData.dataset, epochs=20, steps_per_epoch=40,
                              verbose=1, callbacks = [tboard_callback])
            network.model.save(model)
