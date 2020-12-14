# 编写xml文件程序
'''
制作yolov3训练数据标注xml文件的程序

vesion：20200907
author：Quinlan

update1：0918 之前填写的都是nodle，记得更改为nodule

'''

import xlrd
import xml.dom.minidom

xls_data = xlrd.open_workbook("huizong0907.xls")
table = xls_data.sheet_by_index(0)
dcmnumber = table.col_values(0)
minx = table.col_values(1)
miny = table.col_values(2)
maxx = table.col_values(3)
maxy = table.col_values(4)
minx = [int(x) for x in minx]
miny = [int(x) for x in miny]
maxx = [int(x) for x in maxx]
maxy = [int(x) for x in maxy]
dcmnumber = [int(x) for x in dcmnumber]
dcmnumber = [str(x) for x in dcmnumber]

for num in range(len(dcmnumber)):

#for num in range(10):
    x = minx[num]
    y = miny[num]
    w = maxx[num] - minx[num]
    h = maxy[num] - miny[num]

    doc = xml.dom.minidom.Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    nodefolder = doc.createElement('folder')
    nodefolder.appendChild(doc.createTextNode(str('LIDC')))

    nodefilename = doc.createElement('filename')
    nodefilename.appendChild(doc.createTextNode(str(dcmnumber[num]+'.jpg')))

    nodepath = doc.createElement('path')
    nodepath.appendChild(doc.createTextNode(str('LIDC')))

    nodesource = doc.createElement('source')
    nodedatabase = doc.createElement('databse')
    nodedatabase.appendChild(doc.createTextNode(str('LIDC')))
    nodesource.appendChild(nodedatabase)

    nodesize = doc.createElement('size')
    nodewidth = doc.createElement('width')
    nodeheight = doc.createElement('height')
    nodedepth = doc.createElement('depth')
    nodewidth.appendChild(doc.createTextNode(str('512')))
    nodeheight.appendChild(doc.createTextNode(str('512')))
    nodedepth.appendChild(doc.createTextNode(str('3')))
    nodesize.appendChild(nodewidth)
    nodesize.appendChild(nodeheight)
    nodesize.appendChild(nodedepth)

    nodesegmented = doc.createElement('segmented')
    nodesegmented.appendChild(doc.createTextNode(str('0')))

    nodeobject = doc.createElement('object')
    nodename = doc.createElement('name')
    nodename.appendChild(doc.createTextNode(str('nodle')))
    nodepose = doc.createElement('pose')
    nodepose.appendChild(doc.createTextNode(str('0')))
    nodetruncated = doc.createElement('truncasted')
    nodetruncated.appendChild(doc.createTextNode(str('0')))
    nodedifficult = doc.createElement('difficult')
    nodedifficult.appendChild(doc.createTextNode(str('0')))

    nodebndbox = doc.createElement('bndbox')
    nodexmin = doc.createElement('xmin')
    nodexmin.appendChild(doc.createTextNode(str(minx[num])))
    nodeymin = doc.createElement('ymin')
    nodeymin.appendChild(doc.createTextNode(str(miny[num])))
    nodexmax = doc.createElement('xmax')
    nodexmax.appendChild(doc.createTextNode(str(maxx[num])))
    nodeymax = doc.createElement('ymax')
    nodeymax.appendChild(doc.createTextNode(str(maxy[num])))
    nodebndbox.appendChild(nodexmin)
    nodebndbox.appendChild(nodeymin)
    nodebndbox.appendChild(nodexmax)
    nodebndbox.appendChild(nodeymax)

    nodeobject.appendChild(nodename)
    nodeobject.appendChild(nodepose)
    nodeobject.appendChild(nodetruncated)
    nodeobject.appendChild(nodedifficult)
    nodeobject.appendChild(nodebndbox)

    root.appendChild(nodefolder)
    root.appendChild(nodefilename)
    root.appendChild(nodepath)
    root.appendChild(nodesource)
    root.appendChild(nodesize)
    root.appendChild(nodesegmented)
    root.appendChild(nodeobject)


    fp = open(r'E:\Files\Repositories\\0830\xml\\1214/'+dcmnumber[num].zfill(4)+'.xml', 'w')
    doc.writexml(fp)
