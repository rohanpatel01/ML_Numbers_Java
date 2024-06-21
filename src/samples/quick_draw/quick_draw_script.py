import os
import subprocess
import shutil
import numpy as np
import random;
import gc;

# print(os.getcwd())
# print(os.path.dirname(os.path.realpath(__file__)))

#this script should download np bitmap files for each category, and randomly sample them to be used for training. 
#testcases are stored in 2D array where each row is a 28x28 testcase

def createGSPath(cat):
    return "gs://quickdraw_dataset/full/numpy_bitmap/" + cat + ".npy"

def runCmd(cmd):
    print("Running Command : ", end = "")
    for _ in cmd:
        print(_, end = " ")
    print()

    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout.decode())

def extractTestData(cat, n):    
    gsutil_path = shutil.which("gsutil")    #bruh, doesn't work D:
    gsutil_path = shutil.which('gsutil', path="C:\\Users\\andwe\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin")
    
    #download testcase file
    runCmd([gsutil_path, "-m", "cp", "-r", createGSPath(cat), "."])
    filename = cat + ".npy"

    #extract n testcases randomly
    data = np.load(filename)
    random.shuffle(data)
    cases = data[:n].tolist()

    #make sure to release memory
    del data

    #we're done, delete the file
    os.remove(filename)
    gc.collect()

    return cases

#read categories from text file
categories = []
with open("quick_draw_categories.txt", "r") as file:
    for line in file:
        categories.append(line.strip())

#for each category, randomly extract some test data
N = 1000
test_data = []
for i in range(len(categories)):
    c_test_data = extractTestData(categories[i], N)
    for _ in c_test_data:
        test_data.append([i, _])

#shuffle the tests around, why not
random.shuffle(test_data)

#write test data into text file
filepath = "testing/quick_draw_" + str(N) + "ea.txt"
with open(filepath, "w") as file:
    file.write(str(len(test_data)) + "\n")
    for i in range(len(test_data)):
        file.write(str(test_data[i][0]) + " ")
        for j in range(28 * 28):
            file.write(str(test_data[i][1][j]) + " ")
        file.write("\n")

