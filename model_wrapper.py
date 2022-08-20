# import argparse
# import os
# from numpy import imag
import torchvision.models as models
import torch
import torch.nn as nn
# import torch.utils.data
import numpy as np
from gradcam import GradCAM_model
import torch.nn.functional as F
import json
# from coco_classifier.helper_functions.bn_fusion import fuse_bn_recursively
# from coco_classifier.models import create_model
# from coco_classifier.models.tresnet.tresnet import InplacABN_to_ABN
# from collections import OrderedDict
# import torch.distributed as dist
# import models
# import models.aslloss
# from models.query2label import build_q2l
# from q2l_infer import parser_model_args_model

class Gradient(object):

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

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
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        return gradient



class PytorchModelWrapper_public():
    def __init__(self, model_to_run, img_size, labels_path, device, gradcam_layer=None):
        if model_to_run == 'InceptionV3':
            self.model = models.inception_v3(pretrained=True)
        elif model_to_run == 'GoogleNet':
            self.model = models.googlenet(pretrained=True)
        elif model_to_run == 'ResNet18':
            self.model = models.resnet18(pretrained=True)
        elif model_to_run == 'ResNet152':
            self.model = models.resnet152(pretrained=True)
        elif model_to_run == 'ResNeXt-50-32x4d':
            self.model = models.resnext50_32x4d(pretrained=True)
        elif model_to_run == 'ResNeXt-101-32x8d':
            self.model = models.resnext101_32x8d(pretrained=True)
        elif model_to_run == 'Wide_ResNet-101-2':
            self.model = models.wide_resnet101_2(pretrained=True)
        elif model_to_run == 'Wide_ResNet-50-2':
            self.model = models.wide_resnet50_2(pretrained=True)
        elif model_to_run == 'EfficientNet-B7':
            self.model = models.efficientnet_b7(pretrained=True)
        elif model_to_run == 'regnet_x_32gf':
            self.model = models.regnet_x_32gf(pretrained=True)
        elif model_to_run == 'regnet_y_32gf':
            self.model = models.regnet_y_32gf(pretrained=True)
        elif model_to_run == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(pretrained=True)
        elif model_to_run == 'Xception':
            import timm
            self.model = timm.create_model('xception', pretrained=True)
            #self.model.load_state_dict(torch.load('Imagenet_results/weight.pth'))
        elif model_to_run == 'ResNet50':
            self.model = models.resnet50()
            self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, 20)
            # self.model.load_state_dict(torch.load('pretrained_models/voc_multilabel/resnet50/model-2.pth'))
        elif model_to_run == "Q2L_COCO":
            def clean_state_dict(state_dict):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k[:7] == 'module.':
                        k = k[7:]  # remove `module.`
                    new_state_dict[k] = v
                return new_state_dict

            available_models = ['Q2L-R101-448', 'Q2L-R101-576', 'Q2L-TResL-448', 'Q2L-TResL_22k-448', 'Q2L-SwinL-384',
                                'Q2L-CvT_w24-384']
            parser_model = argparse.ArgumentParser(description='Query2Label for multilabel classification')
            parser_model.add_argument('--dataname', help='dataname', default='coco14', choices=['coco14'])
            parser_model.add_argument('--dataset_dir', help='dir of dataset',
                                      default='/comp_robot/liushilong/data/COCO14/')

            parser_model.add_argument('--img_size', default=448, type=int,
                                      help='image size. default(448)')
            parser_model.add_argument('-a', '--arch', metavar='ARCH', default='Q2L-R101-448',
                                      choices=available_models,
                                      help='model architecture: ' +
                                           ' | '.join(available_models) +
                                           ' (default: Q2L-R101-448)')
            parser_model.add_argument('--config', type=str, help='config file')

            parser_model.add_argument('--output', metavar='DIR',
                                      help='path to output folder')
            parser_model.add_argument('--loss', metavar='LOSS', default='asl',
                                      choices=['asl'],
                                      help='loss functin')
            parser_model.add_argument('--num_class', default=80, type=int,
                                      help="Number of classes.")
            parser_model.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                                      help='number of data loading workers (default: 8)')
            parser_model.add_argument('-b', '--batch-size', default=16, type=int,
                                      metavar='N',
                                      help='mini-batch size (default: 16), this is the total '
                                           'batch size of all GPUs')
            parser_model.add_argument('-p', '--print-freq', default=10, type=int,
                                      metavar='N', help='print frequency (default: 10)')
            parser_model.add_argument('--resume', type=str, metavar='PATH',
                                      help='path to latest checkpoint (default: none)')

            parser_model.add_argument('--pretrained', dest='pretrained', action='store_true',
                                      help='use pre-trained model. default is False. ')

            parser_model.add_argument('--eps', default=1e-5, type=float,
                                      help='eps for focal loss (default: 1e-5)')

            # distribution training
            parser_model.add_argument('--world-size', default=-1, type=int,
                                      help='number of nodes for distributed training')
            parser_model.add_argument('--rank', default=-1, type=int,
                                      help='node rank for distributed training')
            parser_model.add_argument('--dist-url', default='tcp://127.0.0.1:3451', type=str,
                                      help='url used to set up distributed training')
            parser_model.add_argument('--seed', default=None, type=int,
                                      help='seed for initializing training. ')
            parser_model.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
            parser_model.add_argument('--amp', action='store_true',
                                      help='use mixture precision.')
            # data aug
            parser_model.add_argument('--orid_norm', action='store_true', default=False,
                                      help='using oridinary norm of [0,0,0] and [1,1,1] for mean and std.')

            # * Transformer
            parser_model.add_argument('--enc_layers', default=1, type=int,
                                      help="Number of encoding layers in the transformer")
            parser_model.add_argument('--dec_layers', default=2, type=int,
                                      help="Number of decoding layers in the transformer")
            parser_model.add_argument('--dim_feedforward', default=256, type=int,
                                      help="Intermediate size of the feedforward layers in the transformer blocks")
            parser_model.add_argument('--hidden_dim', default=128, type=int,
                                      help="Size of the embeddings (dimension of the transformer)")
            parser_model.add_argument('--dropout', default=0.1, type=float,
                                      help="Dropout applied in the transformer")
            parser_model.add_argument('--nheads', default=4, type=int,
                                      help="Number of attention heads inside the transformer's attentions")
            parser_model.add_argument('--pre_norm', action='store_true')
            parser_model.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                                      help="Type of positional embedding to use on top of the image features")
            parser_model.add_argument('--backbone', default='resnet101', type=str,
                                      help="Name of the convolutional backbone to use")
            parser_model.add_argument('--keep_other_self_attn_dec', action='store_true',
                                      help='keep the other self attention modules in transformer decoders, which will be removed default.')
            parser_model.add_argument('--keep_first_self_attn_dec', action='store_true',
                                      help='keep the first self attention module in transformer decoders, which will be removed default.')
            parser_model.add_argument('--keep_input_proj', action='store_true',
                                      help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
            args_model = parser_model.parse_args([])

            with open("pretrained_models/coco_multilabel/config_new.json", 'r') as f:
                cfg_dict = json.load(f)
            for k, v in cfg_dict.items():
                setattr(args_model, k, v)
            self.model = build_q2l(args_model)
            self.model = self.model.cuda()
            #self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args_model.local_rank], broadcast_buffers=False)
            # criterion = models.aslloss.AsymmetricLossOptimized(
            #     gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
            #     disable_torch_grad_focal_loss=True,
            #     eps=args.eps,
            # )
            checkpoint = torch.load("pretrained_models/coco_multilabel/checkpoint.pkl")
            state_dict = clean_state_dict(checkpoint['state_dict'])
            self.model.load_state_dict(state_dict, strict=True)
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
            self.model.eval()
        elif model_to_run == 'ResNet50_COCO':
            #self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            parser_model = argparse.ArgumentParser()
            parser_model.add_argument('--model_name', default="tresnet_m")
            parser_model.add_argument('--model_path', default="pretrained_models/coco_multilabel/tresnet_m_COCO_224_84_2.pth")
            parser_model.add_argument('--use_ml_decoder', default=1)
            parser_model.add_argument('--image-size', type=int, default=224)
            parser_model.add_argument('--th', type=float, default=0.75)
            parser_model.add_argument('--top-k', type=float, default=20)
            parser_model.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
            parser_model.add_argument('--decoder-embedding', default=768, type=int)
            parser_model.add_argument('--zsl', default=0, type=int)
            parser_model.add_argument('--num_classes', default=91, type=int)

            args_model = parser_model.parse_args([])
            print('creating model {}...'.format(args_model.model_name))
            self.model = create_model(args_model, load_head=True).cuda()
            state = torch.load(args_model.model_path, map_location='cpu')
            self.model.load_state_dict(state['model'], strict=True)
            ########### eliminate BN for faster inference ###########
            self.model = self.model.cpu()
            #self.model = InplacABN_to_ABN(self.model)
            self.model = fuse_bn_recursively(self.model)
            self.model = self.model.cuda().half().eval()
            #######################################################
            '''
            self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, 80)
            self.model.load_state_dict(torch.load('pretrained_models/coco_multilabel/tresnet_m_COCO_224_84_2.pth'))
            '''
            print('done')

        self.device_ = device
        self.model.to(self.device_)
        self.model.eval()
        self.last_conv_layer = self.find_target_layer()
        self.find_layer_idx()
        self.img_size = img_size
        if not gradcam_layer is None:
            self.gradcam_layer = gradcam_layer
        else:
            self.gradcam_layer = self.find_target_layer()
        self.gradCAM_model = GradCAM_model(self.model, img_size, self.device_, gradcam_layer=self.gradcam_layer)
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)


    def preprocess_single_img(self, img):
      x = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # channel first
      x = x[np.newaxis, ...]
      x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
      return x


    def preprocess_imgs(self, imgs):
      x = np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))  # channel first
      x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
      return x


    def get_image_shape(self):
      return (self.img_size,self.img_size)


    def find_target_layer(self):
      layer_name = None
      for name, m in self.model.named_modules():
          if isinstance(m, nn.Conv2d):
              layer_name = name
      if layer_name is None:
          raise ValueError("Could not find conv2d layer. Cannot apply GradCAM")
      return layer_name



    def run_examples(self, images, BOTTLENECK_LAYER):
      if len(images.shape) == 4:
          images = self.preprocess_imgs(images)
      else:
          images = self.preprocess_single_img(images)

      features = []

      def hook(module, input, output):
          features.append(output.clone().detach())

      if BOTTLENECK_LAYER == 'fc':
          handle = self.model.fc.register_forward_hook(hook)
      else:
          #print(self.target_layer_idx)
          handle = self.layers[self.target_layer_idx[BOTTLENECK_LAYER]].register_forward_hook(hook)
      
      #print(images.shape)
      y = self.model(images.to(self.device_))
      handle.remove()
      output = features[0].cpu().data.numpy()
      if len(output.shape) == 4:
          output = np.transpose(output, (0, 2, 3, 1))
      del images
      return output


    # def label_to_id(self, CLASS_NAME):
    #     return self.labels[CLASS_NAME]
    def label_to_id(self, CLASS_NAME):
      return int(self.labels[CLASS_NAME.replace(' ', '_')])


    def get_gradient(self, activations, CLASS_ID, BOTTLENECK_LAYER, x):
      if len(x.shape) == 4:
          x = self.preprocess_imgs(x)
      else:
          x = self.preprocess_single_img(x)

      compute_gradient = Gradient(self.model, BOTTLENECK_LAYER)
      x = x.to(self.device_)
      gradient = compute_gradient(x, CLASS_ID)
      return -1*np.transpose(gradient, (1, 2, 0))


    def find_layer_idx(self):
      self.layers = {}
      self.target_layer_idx = {}
      for idx, (name, module) in enumerate(self.model.named_modules()):
          self.target_layer_idx[name] = idx
          self.layers[idx] = module


    def get_linears(self, x):
      linear_result = self.run_examples(x, 'fc')
      return linear_result


    def get_preds(self, img):
        with torch.no_grad():
            if len(img.shape) == 2:
                img = np.tile(img, [3, 1, 1])
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            x = np.float32(img.copy().transpose(2, 0, 1)) / 255
            x = np.ascontiguousarray(x)
            x = x[np.newaxis, ...]
            x = torch.tensor(x, requires_grad=True)
            try:
                preds = self.model(x)
            except:
                preds = self.model(x.to(self.device_))
            # print(f"preds shape: {preds.shape}")
            preds = F.softmax(preds)
            return img, x, preds

if __name__ == "__main__":
    ResNetWrapper_public()