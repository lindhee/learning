# -*- coding: UTF-8 -*-
# Re-tag the CIFAR 10 dataset to categories cats, dogs and other
# Magnus Lindhé, 2018

CHUNK_SIZE = 1 + 32*32*3 # No of bytes for each image (each pixel has RGB bytes)

fileNames = ["data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"]
inputDir = "/home/lindhe/install/cifar10_data/cifar-10-batches-bin/"
outputDir = "/home/lindhe/install/cifar2_large_data/cifar-10-batches-bin/"

for fileName in fileNames:
    inFileName = inputDir + fileName
    outFileName = outputDir + fileName
    noOfAnimals = 0
    noOfVehicles = 0
    
    with open(outFileName, "wb") as fOut:
        with open(inFileName, "rb") as fIn:
            byte = fIn.read(CHUNK_SIZE)
            while byte != "":
                ba = bytearray(byte)

                # Re-tag animals (except birds and frogs) as (0) and vehicles as (1)
                newTag = 2; # Means remove
                if ba[0]==3 or ba[0]==4 or ba[0]==5 or ba[0]==7:
                    noOfAnimals = noOfAnimals + 1
                    newTag = 0
                elif ba[0]==0 or ba[0]==1 or ba[0]==8 or ba[0]==9:
                    noOfVehicles = noOfVehicles + 1
                    newTag = 1    
                    
                if newTag < 2:
                    ba[0] = newTag
                    fOut.write(ba)
                     
                byte = fIn.read(CHUNK_SIZE)
            print("Filtered " + outFileName + " which contained " + str(noOfAnimals) + " animals and " + str(noOfVehicles) + " vehicles.")
        

