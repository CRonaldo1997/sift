import base64
import requests
from lxml import etree
import cv2
from glob import glob
import pathlib

#host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=EwtZniA3jgPbGo86eT2Tb1t3&client_secret=5iIYPLswLk0hnI3LGoFugsgNFiVmoXjH'
#json = requests.post(host).json()
#print(json)

def write_xml(json,img_path):
    img_arr = cv2.imread(img_path)
    H, W, C = img_arr.shape
    root = etree.Element('annotation')
    folder = etree.SubElement(root, 'folder')
    folder_name = img_path.split('/')[-2]
    folder.text = folder_name
    filename = etree.SubElement(root, 'filename')
    file_name = pathlib.Path(img_path).name
    filename.text = file_name
    path = etree.SubElement(root, 'path')
    path.text = str(img_path)

    source = etree.SubElement(root, 'source')
    database = etree.SubElement(source, 'database')
    database.text = 'Unknown'

    size = etree.SubElement(root, 'size')
    width = etree.SubElement(size, 'width')
    width.text = str(W)
    height = etree.SubElement(size, 'height')
    height.text = str(H)
    depth = etree.SubElement(size, 'depth')
    depth.text = str(C)

    segmented = etree.SubElement(root, 'segmented')
    segmented.text = '0'

    dict_lists = json['words_result']
    for dict_list in dict_lists:
        w = dict_list['location']['width']
        t = dict_list['location']['top']
        l = dict_list['location']['left']
        h = dict_list['location']['height']
        words = dict_list['words']

        object = etree.SubElement(root, 'object')
        name = etree.SubElement(object, 'name')
        name.text = words
        pose = etree.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = etree.SubElement(object, 'truncated')
        truncated.text = '0'
        difficult = etree.SubElement(object, 'difficult')
        difficult.text = '0'
        bndbox = etree.SubElement(object, 'bndbox')
        xmin = etree.SubElement(bndbox, 'xmin')
        xmin.text = str(l)
        ymin = etree.SubElement(bndbox, 'ymin')
        ymin.text = str(t)
        xmax = etree.SubElement(bndbox, 'xmax')
        xmax.text = str(int(l)+int(w))
        ymax = etree.SubElement(bndbox, 'ymax')
        ymax.text = str(int(t)+int(h))
    tree = etree.ElementTree(root)
    tree.write(img_path.replace('jpg','xml'), pretty_print=True, xml_declaration=False, encoding='utf-8')

def main():
    access_token = '24.e84633a89a1ad89063b562924aba61e9.2592000.1563421643.282335-16550167'
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token=' + access_token
    img_dir = '/Users/jiangd001/Documents/Projects/OCR_TOOLs/img_dir/*.jpg'
    for img_path in glob(img_dir):
        f = open(img_path, 'rb')
        img = base64.b64encode(f.read())
        json = requests.post(url, data={'image': img}).json()
        write_xml(json,img_path)

if __name__ == '__main__':
    main()
