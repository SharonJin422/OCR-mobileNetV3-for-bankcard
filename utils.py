import os, sys
import numpy as np
import tensorflow as tf
import random
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import imgaug.augmenters as iaa
from skimage import util
from tools.augment import distort, stretch, perspective
import pdb
nchannels = 1
image_width = 512
image_height = 32
num_features = image_height * nchannels

maxPrintLen = 10
def sobel_edges_detect(grayimage):
    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.medianBlur(grayimage, 3)
    # x = cv2.Sobel(blur_img, cv2.CV_32F, 1, 0, 3)
    # y = cv2.Sobel(blur_img, cv2.CV_32F, 0, 1, 3)
    x = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(blur_img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return sobel_img

def open_dilate_and_erode(gray_img):
    kernel = np.ones((3, 3), np.uint8)
    # 膨胀让轮廓突出
    Dilation = cv2.dilate(gray_img, kernel, iterations=1)
    # cv2.imshow('Dilation ', Dilation)
    # 腐蚀去掉细节
    Erosion = cv2.erode(Dilation, kernel, iterations=1)
    # cv2.imshow('Erosion ', Erosion)
    return Erosion

class DataIterator:
    def __init__(self, data_map):
        data_dir = "/home/yao/OCR/recognition/TextRecognitionTrain/hong_python/hong/card_crop/"
        self.image_names = []
        self.image = []
        self.labels = []
        self.file_lines_list = []

        path_base = '/home/zj/hfw/items/3.line_recognition/kazheng/data/card_10_gt.txt'
        with open(path_base, 'r', encoding='utf-8') as f:
            f_list = f.readlines()
            for file_line in f_list:
                file_line_spline = file_line.strip('\n').split(' ')
                self.image.append(data_dir + file_line_spline[0])
                single_labels = []
                for cha in file_line_spline[1:]:
                    single_labels.append(str(int(cha)))
                self.labels.append(single_labels)


        # path_base2 = "/home/zj/data/kazheng/bankcard/card0/"
        # self.read_gt(path_base2 + 'bankcard_crop/', path_base2 + 'card_gt2_10.txt')
        # print('length is: ', len(self.image))

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])
        return labels

    def get_input_lens(self, sequences):
        # lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
        return sequences

    
    def input_index_generate_batch(self, index=None, dataset_dir=''):
        image_batch = []
        if index:
            for idx, i in enumerate(index):
                image_path = self.image[i]
                # print('image_path is: ', image_path)
                im = cv2.imread(image_path, 0)
                im = open_dilate_and_erode(im)
                im = sobel_edges_detect(im)
                im = preprocess1(im)
                im = self.ratio_maintain_resize(im)
                im = im * 0.003921568627451


                # im = tf.reshape(im, [32, -1, 1])
                # im2 = tf.reshape(im2, [32, -1, 1])
                # im = np.array(im.eval())
                # print(type(im), im.shape, im2.shape)

                # im = np.dstack((np.array(im), np.array(im2)))
                # print(im.shape)
                # im = im / 127.5 - 1.0

                image_batch.append(im)

            label_batch = [self.labels[i] for i in index]
        else:
            # get the whole data as input
            image_batch = self.image
            label_batch = self.labels
        image_batch = np.array(image_batch)
        # image_batch=tf.subtract(tf.divide(image_batch, 127.5), 1.0)
        batch_inputs = self.get_input_lens(np.array(image_batch))
        # batch_inputs,batch_seq_len = pad_input_sequences(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs, batch_labels, label_batch

    def input_index_generate_val_batch(self, index=None, dataset_dir=''):
        image_batch = []
        if index:
            for idx, i in enumerate(index):
                image_path = self.image[i]
                # print('image_path is: ', image_path)
                im = cv2.imread(image_path, 0)
                #                 try:
                #                     im = cv2.imread(image_path, 0)#.astype(np.float32)
                #                 except:
                #                     print(image_path)
                im = open_dilate_and_erode(im)
                im = sobel_edges_detect(im)
                im = preprocess1(im)
                im = self.ratio_maintain_resize(im)
                #                 im = TIA_image_aug(im)
                im = im * 0.003921568627451

                # im = np.dstack((np.array(im), np.array(im2))) #merge into two channels
                # im = im / 127.5 - 1.0
                image_batch.append(im)

            # image_batch=[self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            # get the whole data as input
            image_batch = self.image
            label_batch = self.labels
        image_batch = np.array(image_batch)
        # image_batch=tf.subtract(tf.divide(image_batch, 127.5), 1.0)
        batch_inputs = self.get_input_lens(np.array(image_batch))
        # batch_inputs,batch_seq_len = pad_input_sequences(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)
        # print('label batch is: ', label_batch)
        return batch_inputs, batch_labels, label_batch

    def transdict2(self, dict_path, start, end, ori_txt, to_txt, new_blank=8693):
        # to txt
        dict_path1 = dict_path + 'dict/' + to_txt  # '7356.txt'
        char2num = {}
        num2char = []
        with open(dict_path1, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip('\n')
                # print(line)
                num2char.append(line)
                char2num[str(line)] = i
        # print(len(num2char))
        # ori txt
        dict_path2 = dict_path + 'dict/' + ori_txt  # '5651.txt'
        char2num2 = {}
        num2char2 = []
        with open(dict_path2, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip('\n')
                # print(line)
                num2char2.append(line)
                char2num2[str(line)] = i

        cn_char = '，！？（）／＋﹣＆．：＃％＠＝＊＞＜；［］＼～＄'
        en_char = ',!?()/+-&.:#%@=*><;[]\\~$'
        for i in range(start, end):
            label = self.labels[i]
            for j, cha in enumerate(label):
                try:
                    k = num2char2[int(cha)]
                except:
                    print(label)
                if k in cn_char:
                    idx = cn_char.index(k)
                    k = en_char[idx]
                    # num = char2num[en_char[idx]]
                if k in char2num.keys():
                    kk = char2num[k]
                else:
                    kk = new_blank
                self.labels[i][j] = str(kk)
