import json
import csv
import os
import os.path

# ============ Configurations ============
label_map = {"Car":1,"Truck":2,"Tractor-trailer":3,"Bus":4,"Container":5,"Boat":6,"Plane":7,"Helicopter":8,"Person":9,"Cyclist":10,"Ambiguous":11,"DCR":12}

# root is the path to the dataset
root = "/home/jiasheng/maskrcnn-benchmark/datasets/neo"	
dataset_name = "Neovision2-Training-Heli-001"
# info
description = "Neovision 2 dataset"
url = "http://ilab.usc.edu/neo2/dataset/"
version = "1.0"
year = 2019
contributor = "Prof. Laurent Itti"
date_created = "2019/04/17"

info = dict()
info["description"] = description
info["url"] = url
info["version"] = version
info["year"] = year
info["contributor"] = contributor
info["date_created"] = date_created

# license number
license_url = "http://ilab.usc.edu/neo2/dataset/"
license_id = 99
license_name = "Attribution License"

licenses = list()
license = dict()
license["url"] = license_url
license["id"] = license_id
license["name"] = license_name
licenses.append(license)

ann = dict()
ann["info"] = info
ann["licenses"] = licenses

dataset_id = 1
images_per_dataset = 100000
ann_per_dataset = 10000000

images = list()
image_ids = dict()
for img in os.listdir(os.path.join(root, dataset_name)):
	image = dict()
	index = int(os.path.splitext(img)[0])
	image["id"] = index + images_per_dataset*dataset_id
	image["license"] = license_id
	image["coco_url"] = "Not a coco image"
	image["flickr_url"] = "Dot not have a flickr url"
	image["width"] = 1920	# should be universal size
	image["height"] = 1080
	image["file_name"] = img
	image["date_captured"] = date_created
	images.append(image)
	image_ids[index] = index + images_per_dataset*dataset_id
ann["images"] = images


annotations = list()
i = 0
with open(os.path.join(root, dataset_name + ".csv")) as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for line in reader:
        if line[0] == "Frame" or line[9] == "":
            continue
        annotation = dict()
        img_index = int(line[0])
        x1 = int(line[1])
        y1 = int(line[2])
        x2 = int(line[3])
        y2 = int(line[4])
        x3 = int(line[5])
        y3 = int(line[6])
        x4 = int(line[7])
        y4 = int(line[8])
        iscrowd = 0
        #TODO need to check the correct format for seg
        segmentation = [[x1,y1,x2,y2,x3,y3,x4,y4]]
        #saving bounding box as x y w h
        bbox = [x1,y1,abs(x3-x1),abs(y3-y1)]
        #TODO use all four point to calculate area
        area = abs(x3-x1) * abs(y3-y1)	
        label = line[9]

        i = i + 1
        annotation["id"] = i + dataset_id * ann_per_dataset
        annotation["category_id"] = label_map[label]
        annotation["iscrowd"] = iscrowd
        annotation["segmentation"] = segmentation
        annotation["image_id"] = image_ids[img_index]
        annotation["area"] = area
        annotation["bbox"] = bbox
        annotations.append(annotation)
ann["annotations"] = annotations

categories = list()
for key in label_map:
	category = dict()
	category["supercategory"] = "Heli"
	category["id"] = label_map[key]
	category["name"] = key
	categories.append(category)

ann["categories"] = categories

with open(dataset_name + '.json', 'w') as outfile:
	json.dump(ann, outfile)