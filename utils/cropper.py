# -*- coding:utf-8 -*-
import argparse
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from pathlib import Path

def get_args():
  # base
  parser = argparse.ArgumentParser(description='Crop image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--mode', default='crop',choices=['div', 'crop','box_crop','random_crop'])
  parser.add_argument('--img_id', default='test01')
  parser.add_argument('--target', default='./DSC_0319.JPG')
  parser.add_argument('--target_type',default="file",choices=['file', 'dir'])
  parser.add_argument('--output_path', default='./div/')

  #for divide
  parser.add_argument('--div_w', default=500, type=int,
                        help='image width')
  parser.add_argument('--div_h', default=500, type=int,
                        help='image height')
  #for crop
  parser.add_argument('--startx', default=0, type=int,
                        help='image width')
  parser.add_argument('--starty', default=0, type=int,
                        help='image height')
  parser.add_argument('--endx', default=100, type=int,
                        help='image width')
  parser.add_argument('--endy', default=100, type=int,
                        help='image height')
  #for box crop
  parser.add_argument('--centerx', default=0, type=int,
                        help='image width')
  parser.add_argument('--centery', default=0, type=int,
                        help='image height')
  parser.add_argument('--box_width', default=400, type=int,
                        help='image width')
  parser.add_argument('--box_height', default=300, type=int,
                        help='image height')
  parser.add_argument('--margin_vertical', default=50, type=int,
                        help='image height')
  parser.add_argument('--margin_horizontal', default=50, type=int,
                        help='image height')
  # random crop same with box crop
  parser.add_argument('--total_width', default=400, type=int,
                        help='image width')
  parser.add_argument('--total_height', default=300, type=int,
                        help='image height')
  parser.add_argument('--sample', default=10, type=int,
                        help='image height')

  return parser.parse_args()

def crop(img,startx,starty,endx,endy):
  if len(img.shape)==3:
    sub=img[starty:endy,startx:endx,:]
  else:
    sub=img[starty:endy,startx:endx]
  return sub

def maybe_create(dir):
  path = Path(dir)
  if not path.exists():
    os.makedirs(dir)

#获取某个文件夹下所有文件的名字
def file_name(file_dir): 
    for root, dirs, files in os.walk(file_dir):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        # print(files) #当前路径下所有非目录子文件
        return files

def divide_img(target,target_type,img_id,output_path,div_w=500,div_h=500):
  if target_type == 'file':
    img = [cv2.imread(target)]
  else:
    names = file_name(target)
    img = [cv2.imread('{}{}'.format(target,name)) for name in names]

  #   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  h = img[0].shape[0]
  w = img[0].shape[1]
  n=int(np.floor(h*1.0/div_w))+1
  m=int(np.floor(w*1.0/div_h))+1
  print('h={},w={},n={},m={}'.format(h,w,n,m))
  dis_h=int(np.floor(h/n))
  dis_w=int(np.floor(w/m))
  num=0
  for i in range(n):
    for j in range(m):
      num+=1
      print('i,j=({},{})'.format(i,j))
      for idx in range(len(img)):
        sub=crop(img[idx],dis_w*j,dis_h*i,dis_w*(j+1),dis_h*(i+1)) #img[dis_h*i:dis_h*(i+1),dis_w*j:dis_w*(j+1),:]
        if target_type == 'file':
          maybe_create(output_path)
          cv2.imwrite('{}/'.format(output_path)+'{}_{}_{}.png'.format(img_id,i,j),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else: 
          maybe_create(output_path)
          maybe_create('{}/{}/'.format(output_path,names[idx][:-4]))
          cv2.imwrite('{}/{}/'.format(output_path,names[idx][:-4])+'{}_{}_{}.png'.format(img_id,i,j),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def crop_and_save(target,target_type,img_id,output_path,startx,starty,endx,endy):
  if target_type == 'file':
    img = [cv2.imread(target)]
  else:
    names = file_name(target)
    img = [cv2.imread('{}{}'.format(target,name)) for name in names]
  #print(img)
  #   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  for idx in range(len(img)):
    sub=crop(img[idx],startx,starty,endx,endy) #img[dis_h*i:dis_h*(i+1),dis_w*j:dis_w*(j+1),:]
    #print(sub)
    if target_type == 'file':
      maybe_create(output_path)
      cv2.imwrite(output_path+'{}.png'.format(img_id),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else: 
      maybe_create(output_path)
      maybe_create('{}/{}/'.format(output_path,names[idx][:-4]))
      cv2.imwrite('{}/{}/'.format(output_path,names[idx][:-4])+'{}.png'.format(img_id),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])

  #sub= crop(img,startx,starty,endx,endy)
  #cv2.imwrite(output_path+'{}.png'.format(img_id),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def box_crop(target,target_type,img_id,output_path,centerx,centery,box_width,box_height,margin_vertical,margin_horizontal):
  if target_type == 'file':
    img = [cv2.imread(target)]
  total_w = 2*box_width+2*margin_vertical
  total_h = 2*box_height+2*margin_horizontal
  #left up 
  startx = centerx -(int)(0.5*box_width+margin_vertical)
  starty = centery - (int)(0.5*box_height+margin_horizontal)
  sub= crop(img,startx,starty,startx+total_w,starty+total_h)
  cv2.imwrite(output_path+'{}_leftup.png'.format(img_id),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])

  #left down
  startx = centerx -(int)(0.5*box_width+margin_vertical)
  starty = centery - (int)(1.5*box_height+margin_horizontal)
  sub= crop(img,startx,starty,startx+total_w,starty+total_h)
  cv2.imwrite(output_path+'{}_leftdown.png'.format(img_id),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])

  #right up
  startx = centerx - (int)(1.5*box_width+margin_vertical)
  starty = centery - (int)(0.5*box_height+margin_horizontal)
  sub= crop(img,startx,starty,startx+total_w,starty+total_h)
  cv2.imwrite(output_path+'{}_rightup.png'.format(img_id),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])

  #right down
  startx = centerx - (int)(1.5*box_width+margin_vertical)
  starty = centery - (int)(1.5*box_height+margin_horizontal)
  sub= crop(img,startx,starty,startx+total_w,starty+total_h)
  cv2.imwrite(output_path+'{}_rightdown.png'.format(img_id),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def random_crop(target,target_type,img_id,output_path,centerx,centery,box_width,box_height,total_width,total_height,sample):
  if target_type == 'file':
    img = [cv2.imread(target)]
  else:
    names = file_name(target)
    img = [cv2.imread('{}{}'.format(target,name)) for name in names]

  max_width = total_width-box_width
  max_height =total_height-box_height
  for i in range(sample):
    r_width = random.random()
    r_height = random.random()
    startx =  (int)((centerx-0.5*box_width) - r_width * max_width)
    starty = (int)((centery-0.5*box_height) - r_height * max_height)
    for idx in range(len(img)):
      sub=crop(img[idx],startx,starty,startx+total_width,starty+total_height) #img[dis_h*i:dis_h*(i+1),dis_w*j:dis_w*(j+1),:]
      #print(sub)
      if target_type == 'file':
        maybe_create(output_path)
        cv2.imwrite(output_path+'{}_{}.png'.format(img_id,i),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])
      else: 
        maybe_create(output_path)
        maybe_create('{}/{}/'.format(output_path,names[idx][:-4]))
        cv2.imwrite('{}/{}/'.format(output_path,names[idx][:-4])+'{}_{}.png'.format(img_id,i),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #sub= crop(img,startx,starty,startx+total_width,starty+total_height)
    #cv2.imwrite(output_path+'{}_{}.png'.format(img_id,i),sub, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == '__main__':
  #img_path = './'
  #save_path_list = ['./div/IMG/','./div/GT/']
  #img_list = ['img.png','label.png']  
  args = get_args()
  print("EXEC:{}".format(args))
  if args.mode == 'div': 
    divide_img(args.target,args.target_type,args.img_id,args.output_path,args.div_w,args.div_h)
  elif args.mode == 'crop':
    crop_and_save(args.target,args.target_type,args.img_id,args.output_path,args.startx,args.starty,args.endx,args.endy)
  elif args.mode == 'box_crop':
    box_crop(args.target,args.target_type,args.img_id,args.output_path,args.centerx,args.centery,
    args.box_width,args.box_height,args.margin_vertical,args.margin_horizontal)
  elif args.mode == 'random_crop':
    if args.total_width < args.box_width:
      print('error in args.total_width < args.box_width')
      exit()
    if args.total_height < args.box_height:
      print('error in args.total_height < args.box_height')
      exit() 
    random_crop(args.target,args.target_type,args.img_id,args.output_path,args.centerx,args.centery,
    args.box_width,args.box_height,args.total_width,args.total_height,args.sample)
  else:
    print('uknowned mode')