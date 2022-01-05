#Script to identify overlay screens so that we can remove them from 
# consideration in the tagger

#Author: Michael Curcio

import sys
import shutil
import os
from xml.dom import minidom

rootDir = sys.argv[1]

badFiles = []

for root, dirs, files in os.walk(rootDir):
    for name in files:
        if name[-4:] == '.xml':
            bad = True
            xmldoc = minidom.parse(os.path.join(root, name))
            lst = xmldoc.getElementsByTagName('node')
            for node in lst:
                cl = node.attributes['class'].value
                layout = "Layout" in cl
                view = cl == "android.view.View"
                if not layout and not view:
                    bad = False 

            if bad:
                badFiles.append(os.path.join(root, name))

for item in badFiles:
    dirName = os.path.split(item)[0]
    name = os.path.split(item)[1]
    screenNumber = name[-5]
    parent = os.path.basename(dirName)
    print("screen " + screenNumber + " in application " + dirName)
    
