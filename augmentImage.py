# PURPOSE OF THIS FILE #
# to augment images to imporve accuracy of deep learning model

import os
from PIL import Image

## save cropped image with specified pixels for top, bottom, left and right
# location    : location where train and test images exist
# width       : width of each element in the array
# height      : height of each element in the array
# top         : crop rows at the top N pixels
# bottom      : crop rows at the bottom N pixels
# left        : crop columns at the left N pixels
# right       : crop columns at the right N pixels
# minWidth    : minimum width of cropped image
# minHeight   : minimum height of cropped image
def cropImgs(location, top, bottom, left, right, minWidth, minHeight):
    file_list = os.listdir(location)

    for file in range(len(file_list)): # for each image
        im = Image.open(location + file_list[file])
        width = im.size[0] # width of image
        height = im.size[1] # height of image

        # constraint of minWidth and minHeight
        if width-(left+right) < minWidth or height-(top+bottom) < minHeight: continue

        # crop
        im = im.crop((left, top, width-right, height-bottom))

        # save: XXX.yyy -> XXX(number).yyy (eg: 1_01.png -> 1_0149.png)
        im.save(location + file_list[file].split('.')[0] + str(file) + '.' + file_list[file].split('.')[1])

if __name__ == '__main__':
    cropImgs('images/test/', 4, 4, 4, 4, 16, 24)
