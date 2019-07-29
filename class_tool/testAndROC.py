# -*- coding: utf-8 -*

import torch
import models_lpf.vgg
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import time


start = time.time()


batch_size_set = 64
device = torch.device("cuda")
#model = models_lpf.vgg.vgg16(filter_size=2)
model = models_lpf.vgg.vgg16(filter_size=5)
torch.cuda.set_device(1)

checkpoint=torch.load("/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf5/model_best.pth.tar")
data_dir = "/home2/project_data/BreakPaper/FDY_POY_cam12_inside/"
# model.load_state_dict(checkpoint['state_dict'],strict=False)
model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()},strict=False)
model.to(device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
data_transforms = {
    'val':transforms.Compose([
        transforms.Resize([80,200]),
        transforms.ToTensor(),
        normalize,
        ]),
}

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['val']}
                  
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size_set,
                                             shuffle=False, pin_memory=False)
              for x in ['val']}
              
dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}
class_names = image_datasets['val'].classes

prob_all=np.empty([len(image_datasets['val']),8])
gtlabel_all=np.empty(len(image_datasets['val']),np.int32)
iters = len(image_datasets['val'])/batch_size_set
mod = np.mod(len(image_datasets['val']),batch_size_set) # 余数

if mod!=0:
    iters += 1 # 注意,假如batch不能整除num,最后需要再添加一次

model.eval()
with torch.no_grad():
    idx0=0
    aug=0
    for i, (input, target) in enumerate(dataloaders['val']):
        print('forward process %d/%d'%(i+1,iters))
        #if 0 is not None:
        inputs = input.cuda(1, non_blocking=True)
        # print inputs
        # raw_input()

        target = target.cuda(1, non_blocking=True)
        
        # print "target:",target
        
        #inputs = torch.cat(input, dim=0)   #对input里面的元素横向连接

        idx0 = i*batch_size_set           
        idx1 = inputs.size(0)
        
        probs = torch.nn.Softmax(dim=1)(model(inputs))
        predict = probs.argmax(dim=1).cpu().data.numpy()
        # print "probs:",probs
        # raw_input()
        # print "predict:",predict
        # raw_input()
        
        # print "mid:",(model(inputs))
        # raw_input()
        
        probs = probs.cpu().data.numpy()
        labels = target.cpu().data.numpy()
        
        prob_all[idx0:idx0+idx1] = probs      ###记牢了,python包头不包尾的
        gtlabel_all[idx0:idx0+idx1] = np.array(labels, np.dtype(np.int32))
        # print "all_predict_prob=",prob_all[idx0:idx0+idx1]
        # raw_input()

gtlabel_all[gtlabel_all<3]=0
gtlabel_all[gtlabel_all==3]=1
gtlabel_all[gtlabel_all==4]=1
gtlabel_all[gtlabel_all==5]=0
gtlabel_all[gtlabel_all>5]=1

used_prob = np.empty([len(gtlabel_all),4])
used_prob[:,0:2] = prob_all[:,3:5]
used_prob[:,2:4] = prob_all[:,6:8]

np.save('/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf5/roc/tested_all_prob.npy',prob_all)
np.save('/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf5/roc/gt_label.npy',gtlabel_all)

pos_probs = np.empty(prob_all.shape[0])
for i in range(0,prob_all.shape[0]):
    pos_probi=np.max(used_prob[i,:])#np.max(prob_all[i,4:8])
    pos_probs[i]=pos_probi

np.save('/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf5/roc/tested_used_prob.npy',pos_probs)

print("Done in %.2f s." % (time.time() - start))    

    





