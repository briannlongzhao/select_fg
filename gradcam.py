# import sys
# import argparse
import os
# import json
# sys.path.append(os.getcwd())
# sys.path.insert(0, '../../../')
# import torch.nn as nn
# import torch
# import torchvision.models as models
from matplotlib import pyplot as plt
import cv2
# import random
# from art.estimators.classification import PyTorchClassifier
# from art.attacks.evasion import ProjectedGradientDescent
import numpy as np
# import copy
# import scipy.optimize as opt

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, img_size):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.img_size = img_size

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        #print('output  is :', output)
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        #print('gradient is :', gradient)
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
        #print('feature is :', feature)

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (self.img_size,self.img_size), cv2.INTER_LINEAR)
        return cam


class GradCAM_model:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, img_size, device, gradcam_layer=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.device_ = device
        if not gradcam_layer is None:
            self.layerName = gradcam_layer
        else:
            self.layerName = self.find_target_layer()
        self.grad_cam = GradCAM(self.model, self.layerName, img_size)


    def find_target_layer(self):
        layer_name = None
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_name = name
        if layer_name is None:
            raise ValueError("Could not find conv2d layer. Cannot apply GradCAM")
        return layer_name


    def compute_heatmap(self, x, classIdx, keep_percent=80):
        x = x.to(self.device_)
        #print('class_index', classIdx)
        cam = self.grad_cam(x, classIdx)  # cam mask

        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        #cam = np.where(cam3 > np.median(cam3), 1, 0)
        cam = np.where(cam3 > np.percentile(cam3, 100 - keep_percent), 1, 0)
        return cam3, cam


    def overlay_gradCAM(self, img, cam3, cam):
        new_img = cam * img
        new_img = new_img.astype("uint8")

        cam3 = np.uint8(255 * cam3)
        cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
        new_img_concat = 0.3 * cam3 + 0.5 * img
        new_img_concat = (new_img_concat * 255.0 / new_img_concat.max()).astype("uint8")

        return new_img, new_img_concat

    def find_gradcam_center(self, mask, size):
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] == 0.0:
            M["m00"] = 0.001
        cx = min(M["m01"] / M["m00"], size-1)
        cy = min(M["m10"] / M["m00"], size-1)
        return (cx, cy)

    def showCAMs(self, img, img_id, x, chosen_class, class_name, keep_percent):
        save_dir = "./gradcam_output/"+class_name+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.imshow(img.astype("uint8"))
        plt.axis('off')
        #plt.show()
        #cv2.imwrite(save_dir+img_id.replace(".jpg",'')+'_0.png', img.astype("uint8")[:, :, ::-1])

        cam3, cam = self.compute_heatmap(x=x.cuda(), classIdx=chosen_class, keep_percent=keep_percent)
        new_img, new_img_concat = self.overlay_gradCAM(img, cam3, cam)
        plt.imshow(new_img)
        plt.axis('off')
        #plt.show()

        # Find the center of Gaussian distribution and show in heat map (c1x,c1y)
        '''def gauss2dFunc(xy, xo, yo, sigma):
            x = xy[0]
            y = xy[1]
            return np.exp(-((x - xo) * (x - xo) + (y - yo) * (y - yo)) / (2. * sigma * sigma)).ravel()
        xvec = np.array(range(375))
        yvec = np.array(range(375))
        X, Y = np.meshgrid(xvec, yvec)
        initial_guess = [150, 150, 50]  # x of center, y of center, variance
        try:
            popt, pcov = opt.curve_fit(gauss2dFunc, (X, Y), cam3[:, :, 0].ravel(), p0=initial_guess)
            c1y, c1x = min(round(popt[0]), 298), min(round(popt[1]), 298)
        except:
            c1y, c1x = (148,148)'''
        # Find center of cam mask (c2x,c2y)
        c2 = self.find_gradcam_center(cam[:, :, 0].astype('uint8'), 375)
        c2x,c2y = round(c2[0]),round(c2[1])

        new_img_concat = cv2.cvtColor(new_img_concat, cv2.COLOR_BGR2RGB)
        for i in range(-5,6):
            for j in range(-5,6):
                try:
                    #new_img[c1x+i][c1y+j] = [0, 0, 0]
                    new_img[c2x+i][c2y+j] = [255, 255, 255]
                    #new_img_concat[c1x+i][c1y+j] = [0,0,0]
                    new_img_concat[c2x+i][c2y+j] = [255,255,255]
                except:
                    pass
        plt.imshow(new_img_concat)
        plt.axis('off')
        #plt.show()
        #cv2.imwrite(save_dir+img_id.replace(".jpg",'')+'_1.png', new_img.astype("uint8")[:, :, ::-1])
        #cv2.imwrite(save_dir+img_id.replace(".jpg",'')+'_2.png', new_img_concat.astype("uint8")[:, :, ::-1])


def get_preds(model, IMAGE_PATH='/lab/tmpig23b/u/zhix/VR_SC/ilab2M_pose/train/mil/mil-i0001-b0001-c09-r07-l0-f2.jpg'):
    img = plt.imread(IMAGE_PATH)
    img = cv2.resize(img, (375, 375))
    x = np.float32(img.copy()) / 255
    x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))  # channel first
    x = x[np.newaxis, ...]
    x = torch.tensor(x, requires_grad=True)
    try:
        preds = model(x)
    except:
        preds = model(x.cuda())
    return img, x, preds

