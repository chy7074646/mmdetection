import torch
import mmdet.models.backbones.pvanet as pva
import os

import hiddenlayer as hl

model = pva.PVALiteFeat()

graph=hl.build_graph(model, torch.zeros([1, 3, 224, 224]))

graph.save(os.path.join('./', "model.png"))
