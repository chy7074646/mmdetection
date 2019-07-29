# -*- coding: utf-8 -*

import torch
import models_lpf.vgg
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import matplotlib as mpl
from graphviz import Digraph
from torch.autograd import Variable
from tensorboardX import SummaryWriter

mpl.use('Agg')
import matplotlib.pyplot as plt

def make_dot(var, params=None):
        """
        画出 PyTorch 自动梯度图 autograd graph 的 Graphviz 表示.
        蓝色节点表示有梯度计算的变量Variables;
        橙色节点表示用于 torch.autograd.Function 中的 backward 的张量 Tensors.

        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled', shape='box', align='left',
                                  fontsize='12', ranksep='0.1', height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    # note: this used to show .saved_tensors in pytorch0.2, but stopped
                    # working as it was moved to ATen and Variable-Tensor merged
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    name = param_map[id(u)] if params is not None else ''
                    node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')
                elif var in output_nodes:
                    dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        # 多输出场景 multiple outputs
        if isinstance(var, tuple):
            for v in var:
                add_nodes(v.grad_fn)
        else:
            add_nodes(var.grad_fn)

        #resize_graph(dot)

        return dot

def demo_visual_by_graphviz(model,inputs):
    inputs=inputs.cuda(0, non_blocking=True)
    y = model(inputs)
    g = make_dot(y)
    g.render('test-table.gv.pdf', view=False)
    
def demo_visual_by_tensorboardx(model,inputs):
    dummy_input = torch.rand(13, 3, 80, 200)
    with SummaryWriter(comment='VGG') as w:
        w.add_graph(model, (dummy_input,))
        
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure()
    plt.imshow(inp)
    
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)
    plt.savefig("./imgs.png")    

def validate_shift(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for ep in range(5):
            for i, (input, target) in enumerate(val_loader):
                if 1 is not None:
                    input = input.cuda(1, non_blocking=True)
                target = target.cuda(1, non_blocking=True)

                off0 = np.random.randint(8,size=2)
                off1 = np.random.randint(8,size=2)

                # output0 = model(input[:,:,off0[0]:off0[0]+224,off0[1]:off0[1]+224])
                # output1 = model(input[:,:,off1[0]:off1[0]+224,off1[1]:off1[1]+224])
                output0 = model(input[:,:,off0[0]:off0[0]+72,off0[1]:off0[1]+192])
                output1 = model(input[:,:,off1[0]:off1[0]+72,off1[1]:off1[1]+192])
   
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == '__main__':      
    data_dir = "/home2/project_data/BreakPaper/FDY_POY_cam12_inside/tmp"
    data_dir = '/home2/project_data/BreakPaper/FDY_POY_cam12_inside/'
    start = time.time()

    batch_size_set=1
    device = torch.device("cuda")
    model = models_lpf.vgg.vgg16(filter_size=3)
    torch.cuda.set_device(1)

    checkpoint=torch.load("/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf3/model_best.pth.tar")
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
                      
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=13,
                                                 shuffle=True, pin_memory=True)
                  for x in ['val']}
                  
    dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}
    class_names = image_datasets['val'].classes
    
    


