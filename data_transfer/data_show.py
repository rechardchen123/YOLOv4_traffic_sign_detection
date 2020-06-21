import json
import pylab as pl
import random
import numpy as np
import cv2
import anno_func

datadir = "/home/richardchen123/Documents/data/YOLOv4/data"

filedir = datadir + "/annotations.json"
ids = open(datadir + "/test/ids.txt").read().splitlines()

annos = json.loads(open(filedir).read())

imgid = random.sample(ids, 1)[0]
print(imgid)

imgdata = anno_func.load_img(annos, datadir, imgid)
imgdata_draw = anno_func.draw_all(annos, datadir, imgid, imgdata)
pl.figure(figsize=(20, 20))
pl.imshow(imgdata_draw)



