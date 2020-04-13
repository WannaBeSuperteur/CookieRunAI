# PURPOSE OF THIS FILE #
# to augment images to imporve accuracy of deep learning model

import os
from PIL import Image

## save cropped image with specified pixels for top, bottom, left and right
# location    : location where train and test images exist
# file_list   : list of image files to crop
# width       : width of each element in the array
# height      : height of each element in the array
# top         : crop rows at the top N pixels
# bottom      : crop rows at the bottom N pixels
# left        : crop columns at the left N pixels
# right       : crop columns at the right N pixels
# minWidth    : minimum width of cropped image
# minHeight   : minimum height of cropped image
# i           : changing value to prevent duplicated file name
def cropImgs(location, file_list, top, bottom, left, right, minWidth, minHeight, i):
    for file in range(len(file_list)): # for each image
        im = Image.open(location + file_list[file])
        width = im.size[0] # width of image
        height = im.size[1] # height of image

        # constraint of minWidth and minHeight
        if width-(left+right) < minWidth or height-(top+bottom) < minHeight: continue

        # crop
        im = im.crop((left, top, width-right, height-bottom))

        # save: XXX.yyy -> XXX(number).yyy (eg: 1_01.png -> 1_0149.png)
        im.save(location + file_list[file].split('.')[0] + str(file + i * len(file_list)) +
                '.' + file_list[file].split('.')[1])

## read file and crop images using the content of file
# fileName    : file name
def cropImgsUsingFile(fileName):
    
    f = open(fileName, 'r')
    fl = f.readlines()

    # 1st line indicates location where train and test images exist
    location = fl[0].split('\n')[0]
    file_list = os.listdir(location) # file list
    f.close()

    # 2nd ~ last lines indicate [top, bottom, left, right, minWidth, minHeight]
    # eg: '4 4 4 4 16 24' -> top=4, bottom=4, left=4, right=4, minWidth=16, minHeight=24
    for i in range(1, len(fl)):
        args = fl[i].split('\n')[0].split(' ')
        cropImgs(location, file_list, int(args[0]), int(args[1]), int(args[2]),
                 int(args[3]), int(args[4]), int(args[5]), i-1)

if __name__ == '__main__':
    cropImgsUsingFile('augment.txt')
