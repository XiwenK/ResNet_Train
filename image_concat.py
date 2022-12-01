import os
from math import ceil
from os import listdir
from PIL import Image

# im1 = Image.open(os.getcwd() + '/data characteristics/pic_1_12.png')
# im2 = Image.open(os.getcwd() + '/data characteristics/pic_1_17.png')
#
# width, height = im1.size
# n_width = ceil(width * 0.7)
# n_height = ceil(height * 0.7)
# n_im1 = im1.resize((n_width, n_height))
# n_im2 = im2.resize((n_width, n_height))
# result = Image.new(n_im1.mode, (n_width * 2, n_height * 1))
# result.paste(n_im1, box=(0, 0))
# result.paste(n_im2, box=(n_width, 0))
# result.save(os.getcwd() + '/data characteristics/trainset image.png')
# result.show()

im1 = Image.open(os.getcwd() + '/ResNet Train Figure/ResNet Accuracy 8.png')
im2 = Image.open(os.getcwd() + '/ResNet Train Figure/ResNet Accuracy 9.png')
im3 = Image.open(os.getcwd() + '/ResNet Train Figure/ResNet Accuracy 10.png')
im4 = Image.open(os.getcwd() + '/ResNet Train Figure/ResNet Accuracy 11.png')
acc_ims = [im1, im2, im3, im4]
#
im5 = Image.open(os.getcwd() + '/ResNet Train Figure/ResNet Loss 8.png')
im6 = Image.open(os.getcwd() + '/ResNet Train Figure/ResNet Loss 9.png')
im7 = Image.open(os.getcwd() + '/ResNet Train Figure/ResNet Loss 10.png')
im8 = Image.open(os.getcwd() + '/ResNet Train Figure/ResNet Loss 11.png')
loss_ims = [im5, im6, im7, im8]


# 图片转化为相同的尺寸
# ims = []
# for i in im_list:
#     new_img = i.resize((1280, 1280), Image.BILINEAR)
#     ims.append(new_img)

# 单幅图像尺寸
width, height = im1.size

# 创建空白长图
result = Image.new(im1.mode, (width * 2, height * 4))

# 拼接图片
for i, im in enumerate(acc_ims):
    result.paste(im, box=(0, i * height))

for i, im in enumerate(loss_ims):
    result.paste(im, box=(width, i * height))

result.save(os.getcwd() + '/ResNet Train Figure/lr_data_augmentation.png')
# result.show()
