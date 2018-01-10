# -*- coding: UTF-8 -*-
# Remove all images but cats and dogs in the CIFAR 10 dataset
# Magnus Lindhé, 2018

CHUNK_SIZE = 1 + 32*32*3 # No of bytes for each image (each pixel has RGB bytes)

fileNames = ["data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"]
inputDir = "/tmp/cifar10_data/cifar-10-batches-bin/"
outputDir = "/tmp/cifar10_data/cifar-10-batches-bin/filtered/"

for fileName in fileNames:
    inFileName = inputDir + fileName
    outFileName = outputDir + fileName
    noOfCatsOrDogs = 0
    
    with open(outFileName, "wb") as fOut:
        with open(inFileName, "rb") as fIn:
            byte = fIn.read(CHUNK_SIZE)
            while byte != "":
                ba = bytearray(byte)
                # Only keep cats (3) or dogs (5)
                if ba[0]==3 or ba[0]==5:
                    noOfCatsOrDogs = noOfCatsOrDogs + 1
                    ba[0] = 0 if ba[0]==3 else 1 # Cat = 0, dog = 1
                    fOut.write(ba) 
                byte = fIn.read(CHUNK_SIZE)
            print("Filtered " + outFileName + " which contained " + str(noOfCatsOrDogs) + " cats or dogs.")
        

