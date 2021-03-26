from controller import controller, dataline
if __name__ == "__main__":
    controller = controller('/Data/models/new')

    networks = controller.nextNetwork(controller.archFiles)

    trainData = dataline('/Data/dataset2/', batchMode='sequential')

    batchSize=2


    for network in networks:
        print(network)
        print(network.model.summary())

        network.model.fit(trainData.nextBatch(batchSize), epochs=3, verbose=2)
        network.save()
