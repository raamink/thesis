from controller import controller, dataline
import tensorflow as tf
from datetime import datetime
if __name__ == "__main__":
    controller = controller('/Data/models/new')

    networks = controller.nextNetwork(controller.archFiles)

    trainData = dataline('/Data/dataset2/', batchMode='sequential')

    batchSize=2


    for network in networks:
        print(network)
        print(network.model.summary())

        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

        network.model.fit(trainData.nextBatch(batchSize), epochs=3, steps_per_epoch=40, verbose=2)
        network.save()
