#We clean out the screenshots whos corresponding ux dump contains the phrase
# "package="com.android.launcher", and copy the rest into another directory
#Important: we have taken steps here to preserve the directory structure (we
# keep screens grouped by app)

import sys
import shutil
import os
from xml.dom import minidom

rootDir = sys.argv[1]
outDir = sys.argv[2]

goodFiles = []

for root, dirs, files in os.walk(rootDir):
    for name in files:
        if name[-4:] == '.xml':
            xmldoc = minidom.parse(os.path.join(root, name))
            lst = xmldoc.getElementsByTagName('node')
            for node in lst:
                package = node.attributes['package'].value
                if package == 'com.android.launcher':
                    break
                else:
                    goodFiles.append(os.path.join(root,name)) 
                    correspondingPngName = name[0:-4] + ".png"
                    correspondingPngName = correspondingPngName.replace('hierarchy','screenshot')
                    goodFiles.append(os.path.join(root,correspondingPngName))

for item in goodFiles:
    dirName = os.path.split(item)[0]
    parent = os.path.basename(dirName)
    finalDst = os.path.join(outDir, parent)
    if not os.path.exists(finalDst):
        os.makedirs(finalDst)
    shutil.copy(item, finalDst)
