import torch
import mmdet.models.backbones.pvanet as pva
import mmdet.models.backbones.resnet as res
import mmdet.models.backbones.ssd_vgg as vgg
import mmdet.models.necks.fpn as fpn
import mmdet.models.backbones.resnext as resnext
import os

import hiddenlayer as hl

#model = res.ResNet(18)

#model = pva.PVALiteFeat()

#model = vgg.VGG(16)

model = resnext.ResNeXt(18)

#model=fpn.FPN([256,14,14],[256,14,14],3)

graph=hl.build_graph(model, torch.zeros([1, 3, 224, 224]))

graph.save(os.path.join('./', "model_vgg.png"))
