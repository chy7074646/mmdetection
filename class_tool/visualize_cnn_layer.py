import cv2
import numpy as np
import torch
import models_lpf.vgg
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import os

def get_features(features, layer_number, image):
    x=image
    for index,layer in enumerate(features):
        print "index", index
        raw_input()
        print "layer", layer
        raw_input()
        
        x=layer(x)
        print "xshape:",x.shape
        if(index == layer_number):
            return x

def get_single_feature(features,images):
    
    feature = features[:,0,:,:]
    
    print "sshape:",feature.shape
    
    feature = feature.view(feature.shape[1],feature.shape[2])
    
    return feature
    
def save_feature_to_img(feature):
    feature = feature.cpu().data.numpy()
    feature = 1/(1+np.exp(-1*feature))
    
    feature = np.round(feature*255)
    cv2.imwrite('./feature_img.jpg',feature)
    
def get_parameters(Model,index_number):
    for index,param in enumerate(Model.named_parameters()):
        print "index:",index
        if index==index_number:
           print "cur_params:",param
           return 
      
def save_param_to_txt(Model,save_path):
    output = open(save_path,'w')
    
    for index,param in enumerate(Model.named_parameters()):
        #print "name:",param[0]
        #print "data:",param[1]
        cur_param = param[1].cpu().data.numpy()
        str_cur_param = " ".join(str(i) for i in cur_param)
        #print "data:",str_cur_param
        
        output.write(param[0]+"={"+str_cur_param+"}\n")
        #raw_input()
    output.close()

if __name__=='__main__':
    data_dir = "/home2/project_data/BreakPaper/FDY_POY_cam12_inside/tmp"
    start = time.time()

    batch_size_set=1
    device = torch.device("cuda")
    model = models_lpf.vgg.vgg16(filter_size=3)
    torch.cuda.set_device(1)

    checkpoint=torch.load("/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf3/model_best.pth.tar")
    #model.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(checkpoint['state_dict'])
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

    features=model.features
    print "features:",features
    
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(dataloaders['val']):
            inputs = input.cuda(1, non_blocking=True)
            #features = get_features(features,35,inputs)
            # feature = get_single_feature(features, inputs)
            # save_feature_to_img(feature)
            #get_parameters(model,0)
            #save_param_to_txt(model,'03.txt')
            #np.savetxt('params',aa,delimiter=',')
            
