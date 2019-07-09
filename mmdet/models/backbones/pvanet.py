import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from ..registry import BACKBONES

#from utils import *

class CReLu(nn.Module):
    def __ini__(self, act=F.relu):
        super(CReLu,self).__init__()
        self.act = act
    def forward(self,x):
        x=torch.cat((x,-x),1)
        x=self.act(x)
        
        return x
    
class ConvBnAct(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, **kwargs):
        super(ConvBnAct,self).__init__()
        
        self.conv=nn.Conv2d(n_in, n_out, kernel_size, stride, padding, **kwargs)
        self.bn=nn.BatchNorm2d(n_out)
        self.act=F.relu
        
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)
        
        return x
        

class mCReLU_base(nn.Module):
    def __init__(self,n_in,n_out,kernelsize,stride=1,preAct=False,lastAct=True):
        super(mCReLU_base,self).__init__()
        self._preAct=preAct
        self._lastAct=lastAct
        self.act=F.relu
        
        self.conv3x3=nn.Conv2d(n_in,n_out,kernelsize,stride=stride,padding=kernelsize/2)
        self.bn=nn.BatchNorm2d(n_out*2)
        
    def forward(self,x):
        if self._preAct:
            x=self.act(x)
            
        x=self.conv3x3(x)
        x=torch.cat((x,-x),1)
        x=self.bn(x)
        
        if self._lastAct:
            x=self.act(x)
        
        return x

class mCReLU_residual(nn.Module):
    def __init__(self, n_in, n_red, n_3X3, n_out, kernelsize=3, in_stride=1, proj=False, preAct=False, lastAct=True):
        super(mCReLU_residual,self).__init__()
        self._preAct=preAct
        self._lastAct=lastAct
        self._stride=in_stride
        self.act=F.relu
        
        self.reduce=nn.Conv2d(n_in, n_red, 1, stride=in_stride)
        self.conv3X3=nn.Conv2d(n_red,n_3X3,kernelsize,padding=kernelsize/2)
        self.bn=nn.BatchNorm2d(n_3X3*2)
        self.expand=nn.Conv2d(n_3x3*2,n_out,1)
        
        if in_stride>1:
            assert(proj)
        
        self.proj=nn.Conv2d(n_in, n_out, 1, stride=in_stride) if proj else None
        
    def forward(self,x):
        x_sc= x
        
        if self._preAct:
            x = self.act(x)
        
        # Conv 1x1 - Relu
        x = self.reduce(x)
        x = self.act(x)
        
         # Conv 3x3 - mCReLU (w/ BN)
        x = self.conv3X3(x)
        x = torch.cat((x,-x),1)
        x = self.bn(x)
        x = self.act(x)
        
        # Conv 1x1
        x = self.expand(x)
        
        if self._lastAct:
            x = slef.act(x)
         
        #Projection
        if self.proj:
            x_sc = self.proj(x_sc)
        
        x = x + x_sc
        
        return x
    
class Inception(nn.Module):
    def __init__(self,n_in, n_out, in_stride=1, preAct=False, lastAct=True, proj=False,lastConv=True):
        super(Inception,self).__init__()
        
        self._preAct=preAct
        self._lasAct=lastAct
        self.n_in=n_in
        self.n_out=n_out
        self.act_func=nn.ReLU
        self.act=F.relu
        self.in_stride=in_stride
        self.lastConv=lastConv
        
        self.n_branches=0
        self.n_outs=[]
        
        self.proj=nn.Conv2d(n_in, n_out, 1, stride=in_stride) if proj else None
        
    def add_branch(self,module,n_out):
        br_name='branch_{}'.format(self.n_branches)
        setattr(self,br_name,module)
        
        self.n_outs.append(n_out)
        self.n_branches+=1
    
    def branch(self,idx):
        br_name='branch_{}'.format(idx)
        return getattr(self,br_name,None)
        
    def add_convs(self,n_kernels,n_chns):
        assert(len(n_kernels)==len(n_chns))
        
        n_last=self.n_in
        layers=[]
        
        stride=-1
        for k,n_out in zip(n_kernels,n_chns):
            
            if stride==-1:
                stride=self.in_stride
                
            else:
                stride=1
            
            #initialize params
            conv=nn.Conv2d(n_last, n_out, kernel_size=k, bias=False, padding=int(k/2), stride=stride)
            bn=nn.BatchNorm2d(n_out)
            
            layers.append(conv)
            layers.append(bn)
            layers.append(self.act_func())
            
            n_last=n_out
            
        self.add_branch(nn.Sequential(*layers),n_last)
        
        return self
    
    def add_poolconv(self,kernel,n_out,type='MAX'):
        assert(type in ['AVE', 'MAX'])
        n_last=self.n_in
        
        layers=[]
        
        #Pooling
        if type=='MAX':
            layers.append(nn.MaxPool2d(kernel,padding=int(kernel/2),stride=self.in_stride))
        elif type=='AVE':
            layers.append(nn.AvgPool2d(kernel,padding=int(kernel/2),stride=self.in_stride))
        
        # Conv - BN - Act
        layers.append(nn.Conv2d(n_last,n_out,kernel_size=1))
        layers.append(nn.BatchNorm2d(n_out))
        layers.append(self.act_func())
        
        self.add_branch(nn.Sequential(*layers),n_out)
        
        return self
    
    def finalize(self):
        #add 1*1 convolution
        total_outs=sum(self.n_outs)
        
        self.last_conv=nn.Conv2d(total_outs,self.n_out,kernel_size=1)
        self.last_bn=nn.BatchNorm2d(self.n_out)
        
        return self
        
    
    def forward(self,x):
        x_sc=x
        
        if(self._preAct):
            x=self.act(x)
        
        #compute branch

        h=[]
        for i in range(self.n_branches):
            module=self.branch(i)
            assert(module!=None)
            
            h.append(module(x))
            
        x=torch.cat(h,dim=1)
        
        if (self.lastConv):
            x=self.last_conv(x)
            x=self.last_bn(x)
        
        if(self._lasAct):
            x=self.act(x)
            
        if(x_sc.get_device() != x.get_device()):
            print("Something is wrong")
        
        #Projection
        if self.proj:
            x_sc=self.proj(x_sc)
            x=x+x_sc

        return x

# This class is impl. separately so that we can modify feature extraction codes for OD models
# (e.g. concatenating three intermediate outputs at different scales)
class PVANetFeat(nn.Module):
    #this class is im
    def __init__(self):
                 
        super(PVANetFeat,self).__init__()
        
        self.conv1=nn.Sequential(
                mCReLU_base(3,16,kernelsize=7,stride=2,lastAct=False),
                nn.MaxPool2d(3,padding=1,stride=2)
        )
        
        #1/4
        self.conv2=nn.Sequential(
                mCReLU_residual(32,24,24,64,kernelsize=3,preAct=True,lastAct=False,in_stride=1,proj=True),
                mCReLU_residual(64,24,24,64,kernelsize=3,preAct=True,lastAct=False),
                mCReLU_residual(64,24,24,64,kernelsize=3,preAct=True,lastAct=False)
                )
        
        #1/8
        self.conv3=nn.Sequential(
                mCReLU_residual(64,48,48,128,kernelsize=3,preAct=True,lastAct=False,in_stride=2,proj=True),
                mCReLU_residual(128,48,48,128,kernelsize=3,preAct=True,lastAct=False),
                mCReLU_residual(128,48,48,128,kernelsize=3,preAct=True,lastAct=False),
                mCReLU_residual(128,48,48,128,kernelsize=3,preAct=True,lastAct=False)
                )
        
        #1/16
        self.conv4=nn.Sequential(
                self.gen_InceptionA(128, 2, True),
                self.gen_InceptionA(256, 1, False),
                self.gen_InceptionA(256, 1, False),
                self.gen_InceptionA(256, 1, False)
                )
        
        #1/32
        self.conv5 = nn.Sequential(
                self.gen_InceptionB(256, 2, True),
                self.gen_InceptionB(384, 1, False),
                self.gen_InceptionB(384, 1, False),
                self.gen_InceptionB(384, 1, False),

                nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)         # 1/4 feature
        x2 = self.conv3(x1)         # 1/8
        x3 = self.conv4(x2)         # 1/16
        x4 = self.conv5(x3)         # 1/32

        return x4
    
        
    def gen_InceptionA(self,n_in,stride=1,poolconv=False,n_out=256):
            
        if(n_in != n_out) or (stride>1):
            proj=True
        else:
            proj=False
        
        module=Inception(n_in,n_out,preAct=True,lastAct=False,in_stride=stride,proj=proj)\
                .add_convs([1],[64])\
                .add_convs([1,3],[48,128])\
                .add_convs([1,3,3],[24,48,48])
        
        if poolconv:
            module.add_poolconv(3,128)
            
        return module.finalize()
    
        
    def gen_InceptionB(self,n_in,stride=1,poolconv=False,n_out=384):
        
        if(n_in!=n_out) or (stride>1):
            proj=True
        else:
            proj=False
        
        module=Inception(n_in,n_out,preAct=True,lastAct=False,in_stride=stride,proj=proj)\
                .add_convs([1],[64])\
                .add_convs([1,3],[96,192])\
                .add_convs([1,3,3],[32,64,64])
        
        if poolconv:
            module.add_poolconv(3,128)
        
        return module.finalize()
    
@BACKBONES.register_module        
class PVALiteFeat(nn.Module):
    def __init__(self,
                 num_stages=5,
                 out_indices=(0, 1, 2, 3, 4),
                 norm_eval=True):
                 
        super(PVALiteFeat,self).__init__()
        
        self.num_stages = num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.norm_eval = norm_eval
        self.act=F.relu
        
        self.conv1=nn.Sequential(ConvBnAct(3,32,4,2,1))
        self.conv2=nn.Sequential(ConvBnAct(32,48,3,2,1))
        self.conv3=nn.Sequential(ConvBnAct(48,96,3,2,1))
        
        self.incep3=nn.Sequential(
                self.gen_InceptionLiteA(96,2,True),
                self.gen_InceptionLiteA(192,1,False),
                self.gen_InceptionLiteA(192,1,False),
                self.gen_InceptionLiteA(192,1,False),
                self.gen_InceptionLiteA(192,1,False)  
                
                )
        
        self.incep4=nn.Sequential(
                self.gen_InceptionLiteB(192,2,True),
                self.gen_InceptionLiteB(256,1,False),
                self.gen_InceptionLiteB(256,1,False),
                self.gen_InceptionLiteB(256,1,False),
                self.gen_InceptionLiteB(256,1,False)
                
                )
                
        #upsample
        self.upsample=nn.Sequential(
                #nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True)
                nn.ConvTranspose2d(256,256,4,2,1,groups=256,bias=False)
               )
        
        self.downsample=nn.Sequential(
                nn.MaxPool2d(3,2,1)  #achieve downsampling by pool,note caffe (k=3,s=2,p=0)
               )
        
        self.convf=nn.Sequential(
                nn.Conv2d(544,256,1,1)
               )    
       
    def forward(self,x):
        
        #stage 0
        x1=self.conv1(x)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        
        outs = []
        if 0 in self.out_indices:
           outs.append(x3) 
        
        #stage 1
        Ix3=self.incep3(x3)
        if 1 in self.out_indices:
           outs.append(Ix3)
        
        #stage 2
        Ix4=self.incep4(Ix3)
        if 2 in self.out_indices:
           outs.append(Ix4)
        
        #stage 3
        x5_1=self.upsample(Ix4)
        if 3 in self.out_indices:
           outs.append(x5_1)
        
        x5_0=self.downsample(x3)
        
        #stage 4
        x6=torch.cat([x5_0,torch.cat([x5_1,Ix3],dim=1)],dim=1)
        x6=self.convf(x6)
        x6=self.act(x6)
        if 4 in self.out_indices:
           outs.append(x6)
        
        #out=self.convf(x6)
        
        #print("------------pva_lite_feat---------",out.shape)
        
        return tuple(outs)
        
           
    def gen_InceptionLiteA(self,n_in,stride=1,poolconv=False,n_out=192):
        
        module=Inception(n_in,n_out,preAct=False,lastAct=False,in_stride=stride,proj=False,lastConv=False)\
                .add_convs([1,3,3],[16,32,32])\
                .add_convs([1,3],[16,64])
                
        if poolconv:
            return module.add_poolconv(3,96)       
            
        return module.add_convs([1],[96])
        
        
    def gen_InceptionLiteB(self,n_in,stride=1,poolconv=False,n_out=256):
        
        module=Inception(n_in,n_out,preAct=False,lastAct=False,in_stride=stride,proj=False,lastConv=False)\
                .add_convs([1,3,3],[16,32,32])\
                .add_convs([1,3],[32,96])
        
        if poolconv:
            return module.add_poolconv(3,128)
        
        return module.add_convs([1],[128])
        
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                        # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    else:
                        print('Not initializing')
                elif isinstance(m, _BatchNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                    
        else:
            raise TypeError('pretrained must be a str or None')
    
    def train(self, mode=True):
        super(PVALiteFeat, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

            
class PVANet(nn.Module):
    def __init__(self,inputsize=224,num_classes=1000):
        super(PVANet,self).__init__()
        
        self.features=PVANetFeat()
        
        #make sure that input size is multiplier by 32
        assert(inputsize%32==0)
        featsize=inputsize/32
        
        self.classifier=nn.Sequential(
                nn.Linear(384*featsize*featsize,4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                
                nn.Linear(4096,4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inpalce=True),
                nn.Dropout(),
                
                nn.Linear(4096, num_classes)
                )
        
        #Initialize all vars
        initvars(self.modules())
        
    def forward(self,x):
        x=slef.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        
        return x

class PVALiteNet(nn.Module):
    
    def __init__(self,inputsize=224,num_classes=1000):
        super(PVALiteNet,self).__init__()
        
        self.features=PVALiteFeat()
        
        #make sure that input size is multiplier by 32
        assert(inputsize%32==0)
        featsize=inputsize/32*2 ##because it has a upsample layer
        tmp=int(256*featsize*featsize)
        
        self.classifier=nn.Sequential(
                nn.Linear(tmp,4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                
                nn.Linear(4096,4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                
                nn.Linear(4096, num_classes)
                )
        
        #Initialize all vars
        #initvars(self.modules())
    
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        
        return x
        
# def pvanet(**kwargs):
    # #model=PVANet(**kwargs)
    # model=PVALiteNet(**kwargs)
    
    # return model
        