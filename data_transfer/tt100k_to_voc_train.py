import json
import os,sys,shutil
from PIL import Image
from voc_xml_generator import xml_fill

def find_image_size(filename):
    with Image.open(filename) as img:
        img_width = img.size[0]
        img_height = img.size[1]
        img_mode = img.mode
        if img_mode == "RGB":
            img_depth = 3
        elif img_mode == "RGBA":
            img_depth = 3
        elif img_mode == "L":
            img_depth = 1
        else:
            print("img_mode = %s is neither RGB or L" % img_mode)
            exit(0)

        return img_width, img_height, img_depth

def load_mask(annos, datadir, imgid, filler):
    img = annos["imgs"][imgid]
    path = img['path']
    for obj in img['objects']:
        name = obj['category']
        box = obj['bbox']
        xmin = int(box['xmin'])
        ymin = int(box['ymin'])
        xmax = int(box['xmax'])
        ymax = int(box['ymax'])
        filler.add_obj_box(name, xmin, ymin, xmax, ymax)

tt100k_parent_dir = "/home/richardchen123/Documents/data/YOLOv4_data/"

work_sapce_dir = os.path.join(tt100k_parent_dir, "VOCdevkit/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
work_sapce_dir = os.path.join(work_sapce_dir, "VOC2007/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
jpeg_images_path = os.path.join(work_sapce_dir, 'JPEGImages')
annotations_path = os.path.join(work_sapce_dir, 'Annotations')
if not os.path.isdir(jpeg_images_path):
    os.mkdir(jpeg_images_path)
if not os.path.isdir(annotations_path):
        os.mkdir(annotations_path)

datadir = "/home/richardchen123/Documents/data/YOLOv4/data"
filedir = datadir + "/annotations.json"
ids = open(datadir + "/train/ids.txt").read().splitlines()
annos = json.loads(open(filedir).read())

for i, value in enumerate(ids):
    imgid = value
    filename = datadir + "/train/" + imgid + ".jpg"
    width,height,depth = find_image_size(filename)
    filler = xml_fill(filename, width, height, depth)
    load_mask(annos, datadir, imgid, filler)
    filler.save_xml(annotations_path + '/' + imgid + '.xml')
    print("%s.xml saved\n"%imgid)
