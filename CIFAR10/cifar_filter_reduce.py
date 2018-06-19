# -*- coding: UTF-8 -*-
# Re-tag the CIFAR 10 dataset to categories cats, dogs and other
# Magnus Lindhé, 2018

CHUNK_SIZE = 1 + 32*32*3 # No of bytes for each image (each pixel has RGB bytes)

fileNames = ["data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"]
inputDir = "/home/lindhe/install/cifar10_data/cifar-10-batches-bin/"
outputDir = "/home/lindhe/install/cifar3_data/cifar-10-batches-bin/"

for fileName in fileNames:
    inFileName = inputDir + fileName
    outFileName = outputDir + fileName
    noOfCatsOrDogs = 0
    
    with open(outFileName, "wb") as fOut:
        with open(inFileName, "rb") as fIn:
            byte = fIn.read(CHUNK_SIZE)
            while byte != "":
                ba = bytearray(byte)

                # Re-tag cats (3), dogs (5) and other to cat (0), dog (1) and other (2)
                if ba[0]==3 or ba[0]==5:
                    noOfCatsOrDogs = noOfCatsOrDogs + 1
                    
                if ba[0]==3:
                	ba[0]=0
                elif ba[0]==5:
                	ba[0]=1
                else:
                	ba[0]=2
                fOut.write(ba) 
                byte = fIn.read(CHUNK_SIZE)
            print("Filtered " + outFileName + " which contained " + str(noOfCatsOrDogs) + " cats or dogs.")
        

