import os
import sys
import re
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


""" compute bleu by evenly waiting all N-gram bleu scores"""
def computeBleu(bleu4, bleu3, bleu2, bleu1):
    return (float(bleu4) + float(bleu3) + float(bleu2) + float(bleu1)) / 4

def buildDataMat(f):
    mat = []
    readingFile = open(f, 'r')
    count = 0
    for line in readingFile:
        count += 1
        if count == 1:
            continue
        curDict = {}
        curArr = line.split(',')
        curDict['iter'] = int(curArr[0])
        curDict['CIDEr'] = float(curArr[1])
        curDict['BLEU'] = computeBleu(curArr[2], curArr[3], curArr[4], curArr[5])
        curDict['ROUGE_L'] = float(curArr[6])
        curDict['METEOR'] = float(curArr[7])
        mat.append(curDict)
    return mat

def buildKeyData(fi):
    keyData = {}
    f = open(fi, 'r')
    count = 0
    for line in f:
        count += 1
        if count == 1:
            continue
        curArr = line.rstrip().split(",")
        curDict = {}
        iteration = curArr[0]
        curDict['rnnsize'] = int(curArr[1])
        curDict['encodingsize'] = int(curArr[2])
        curDict['lmdroprate'] = float(curArr[3])
        curDict['optalgo'] = curArr[4]
        curDict['lmlearnrate'] = float(curArr[5])
        curDict['decayinterval'] = int(curArr[6])
        curDict['decaybegin'] = int(curArr[7])
        curDict['cnnlearnrate'] = float(curArr[8])
        keyData[iteration] = curDict
    return keyData

def buildKeySequences(keyData, metrics):
    sequences = {}
    #build up the empty lists
    for params in keyData['0'].keys():
        sequences[params] = []
    oneTrial = metrics['0']
    oneDataDict = oneTrial[0]
    for m in oneDataDict.keys():
        sequences[m] = []
    #build sequences to construct scatterplots
    for key in keyData.keys():
        curTrial = keyData[key]
        for metric in curTrial.keys():
            #print('metric: ' + metric)
            metricVal = curTrial[metric] 
            for dics in metrics[key]:
                sequences[metric].append(metricVal)
        for dics in metrics[key]:
            print('dics: ' + str(dics))
            print('metric: ' + metric)
            for d in dics.keys():
                sequences[d].append(dics[d])
    #debugging
    for k in sequences.keys():
        print(k + ':' + str(len(sequences[k])))
    return sequences

def buildScatterPlot(seq, metric, param, path):
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("iterations")
    ax.set_ylabel(param)
    ax.set_zlabel(metric)
    ax.scatter(seq['iter'], seq[param], seq[metric], c=seq[metric], cmap='hot')

    fileName = param + '.png'
    fileDir = os.path.join(path, metric)
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
    filePath = os.path.join(fileDir, fileName)
    fig.savefig(filePath)
    pyplot.close('all')

def buildScatterDir(seq, metric, path):
    params = ['rnnsize', 'encodingsize', 'lmdroprate', 
            'lmlearnrate', 'decayinterval', 'decaybegin', 'cnnlearnrate']
    for param in params:
        buildScatterPlot(seq, metric, param, path)
        
def buildAllScatterPlots(seq, path):
    buildScatterDir(seq, 'BLEU', path) 
    buildScatterDir(seq, 'CIDEr', path)
    buildScatterDir(seq, 'METEOR', path)
    buildScatterDir(seq, 'ROUGE_L', path)
                   
dataDir = sys.argv[1]
figuresDir = sys.argv[2]

dataDict = {}
for (dirpath, dirnames, filenames) in os.walk(dataDir):
    for f in filenames:
        if 'key' not in f:
            #print(f)
            regex = re.compile(r'\d+')
            trialNumber = regex.search(f).group() 
            #print(trialNumber)
            dataDict[trialNumber] = buildDataMat(os.path.join(dirpath, f))
        else:
            keyFile = os.path.join(dirpath, f)

print(keyFile)
keyData = buildKeyData(keyFile)
sequences = buildKeySequences(keyData, dataDict)
buildAllScatterPlots(sequences, figuresDir)
