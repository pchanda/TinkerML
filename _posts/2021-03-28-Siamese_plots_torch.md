---
layout: post
title: Siamese Network using Pytorch with simulated scatter plot data.
categories: ['DeepLearning','Pytorch']
---


[FullNotebook for this post](https://github.com/pchanda/pchanda.github.io/blob/master/data/Siamese_plots.ipynb)

[FullNotebook for the same using torch dataloader](https://github.com/pchanda/pchanda.github.io/blob/master/data/Siamese_plots_with_dataloader.ipynb)

Both the notebooks use the following data: [DataUsed](https://github.com/pchanda/pchanda.github.io/tree/master/data/mol_data)

The same should also work with a smaller version of MNIST data, see ["MnistNotebook"](https://github.com/pchanda/pchanda.github.io/blob/master/data/Siamese_mnist.ipynb)
and data ["MNISTSmall"](https://github.com/pchanda/pchanda.github.io/blob/master/data/mnist_small.tar.gz)


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
import glob
from torchviz import *
```


```python
POS_LABEL = 0 # Pair of Images that match
NEG_LABEL = 1 # Pair of Images that do not match
#If you reverse the labels, you have to change the Contrastive Loss function.
SZ = 128
MARGIN = 5.0
```

## Model


```python
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*SZ*SZ, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def feature_extract(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)        
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.feature_extract(input1) #extract features from image0
        output2 = self.feature_extract(input2) #extract features from image1
        return output1, output2
    
    
net = SiameseNetwork()
print(net)

X0 = torch.zeros((2,1, SZ, SZ)) #channel first (after batch)
X1 = torch.zeros((2,1, SZ, SZ)) #channel first (after batch)
d1,d2 = net(X0,X1)
```

    SiameseNetwork(
      (cnn1): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1))
        (2): ReLU(inplace=True)
        (3): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1))
        (6): ReLU(inplace=True)
        (7): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReflectionPad2d((1, 1, 1, 1))
        (9): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1))
        (10): ReLU(inplace=True)
        (11): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (fc1): Sequential(
        (0): Linear(in_features=131072, out_features=500, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=500, out_features=500, bias=True)
        (3): ReLU(inplace=True)
        (4): Linear(in_features=500, out_features=5, bias=True)
      )
    )


## Contrastive Loss function


```python
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        pred = (self.margin < euclidean_distance).type(torch.float)         
        return loss_contrastive, euclidean_distance, pred
```

## BCELoss (did not do well)


```python
#did not work well, you can try by setting criterion = SimpleBCELoss() below.
class SimpleBCELoss(torch.nn.Module):
    
    def __init__(self):
        super(SimpleBCELoss,self).__init__()
        self.bce_loss = nn.BCELoss()
            
    def forward(self,output1,output2,label):
        edist = nn.PairwiseDistance(p=2,keepdim=True)(output1,output2)
        edist = torch.sigmoid(edist)
        loss_bce = self.bce_loss(edist,label)
        return loss_bce
```

## Make paired data


```python
def get_positive_pairs(path='./data/mol_data/*'): #both images of same digit
    positive_pairs = []
    all_fam_dirs = glob.glob(path)
    for famdir in all_fam_dirs:
        mol_files = glob.glob(famdir+'/*.png')
        for ff1 in mol_files:
            for ff2 in mol_files:
                if ff1 < ff2:
                    positive_pairs.append((ff1,ff2))
    return positive_pairs


def get_negative_pairs(path='./data/mol_data/*',cnt=100): #images are from different digits
    negative_pairs = []
    all_fam_dirs = glob.glob(path)
    random.shuffle(all_fam_dirs)
    all_fam_dirs_rev = all_fam_dirs[::-1] #reversed
    for famdir1,famdir2 in zip(all_fam_dirs,all_fam_dirs_rev):
        if famdir1!=famdir2:
            mol_files_1 = glob.glob(famdir1+'/*.png')
            mol_files_2 = glob.glob(famdir2+'/*.png')
            for ff1 in mol_files_1:
                for ff2 in mol_files_2:
                    negative_pairs.append((ff1,ff2))
        if len(negative_pairs) >= cnt:
            break
            
    return negative_pairs


def read_img(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img = img.resize((SZ,SZ))
    img = np.asarray(img,dtype=np.float32)/255.0
    return img

def build_paired_data(path,shuffle):
    positive_pairs = get_positive_pairs(path) 
    negative_pairs = get_negative_pairs(path,len(positive_pairs))
    
    print('Got ',len(positive_pairs),'positive_pairs')
    print('Got ',len(negative_pairs),'negative_pairs')
    
    if shuffle:
        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)
        
    positive_labels = [POS_LABEL]*len(positive_pairs)
    negative_labels = [NEG_LABEL]*len(negative_pairs)

    all_pairs = positive_pairs + negative_pairs
    all_labels = positive_labels + negative_labels
 
    data = list(zip(all_pairs,all_labels))
    random.shuffle(data)
    print('Loading data size',len(data))
    
    pairImages = []
    pairLabels = []
    pairNames = []

    for image_pair,label in data:
        img0 = read_img(image_pair[0])
        img1 = read_img(image_pair[1])
        
        pairImages.append([img0,img1])
        pairLabels.append([label]) #very important to have labels as shape `batch_size` x 1   
        pairNames.append([image_pair[0],image_pair[1]])
        
    return np.expand_dims(np.array(pairImages),axis=2), np.array(pairLabels), np.array(pairNames)


pairTrain, labelTrain, pairNames = build_paired_data('./data/mol_data/*',True)
print(pairTrain.shape, labelTrain.shape)
```

    Got  200 positive_pairs
    Got  200 negative_pairs
    Loading data size 400
    (400, 2, 1, 128, 128) (400, 1)


## Create Siamese network and train


```python
new_net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(new_net.parameters(),lr = 0.0005)

num_epochs = 10
batch_size = 64

num_batches = len(pairTrain) // batch_size

counter = []
loss_history = [] 
itr_no = 0

for epoch in range(num_epochs):
    epoch_loss = []  # Sum of training loss, no. of tokens
    epoch_accuracy = []
    
    for batch_no in range(num_batches):

        optimizer.zero_grad()

        # Local batches and labels
        X =  pairTrain[batch_no*batch_size:(batch_no+1)*batch_size,]
        y = labelTrain[batch_no*batch_size:(batch_no+1)*batch_size,]

        X0 = torch.tensor(X[:, 0]).float().cuda()
        X1 = torch.tensor(X[:, 1]).float().cuda()
        Y = torch.tensor(y).float().cuda()
        
        output1,output2 = new_net(X0,X1)
        loss_contrastive, edist, predictions = criterion(output1,output2,Y)
        loss_contrastive.backward()
        optimizer.step()

        epoch_loss.append(loss_contrastive.item())
        acc = (Y==predictions).type(torch.float).cpu().numpy()
        epoch_accuracy.extend(acc)

    epoch_loss = np.mean(epoch_loss)
    print('epoch',epoch,'loss=',epoch_loss,'acc=',np.mean(epoch_accuracy))
    loss_history.append(epoch_loss)
    counter.append(epoch)
```

    epoch 0 loss= 8.871821959813436 acc= 0.8203125
    epoch 1 loss= 4.87911335627238 acc= 0.9505208
    epoch 2 loss= 3.5122199058532715 acc= 0.9244792
    epoch 3 loss= 0.9253520170847574 acc= 0.9661458
    epoch 4 loss= 0.33208129554986954 acc= 0.984375
    epoch 5 loss= 0.1407537336150805 acc= 0.9895833
    epoch 6 loss= 0.07411744073033333 acc= 1.0
    epoch 7 loss= 0.033875590190291405 acc= 0.9895833
    epoch 8 loss= 0.019601694618662197 acc= 1.0
    epoch 9 loss= 0.015237660147249699 acc= 1.0


## Plot training loss


```python
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
show_plot(counter,loss_history)
```


    
![an image alt text]({{ site.baseurl }}/images/ooutput_14_0.png "xxx"){:height="40%" width="40%"}    


## Visualize dynamic graph


```python
make_dot(loss_contrastive)
```


![an image alt text]({{ site.baseurl }}/images/oooutput_16_1.svg "xxx"){:height="100%" width="100%"} |    



## Plot some images and their predictions


```python
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(280, 10, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    
```


```python
X =  pairTrain[100:110,]
y = labelTrain[100:110,]

X0 = torch.tensor(X[:, 0]).float().cuda()
X1 = torch.tensor(X[:, 1]).float().cuda()
Y = torch.tensor(y).float().cuda()
        
output1,output2 = new_net(X0,X1)
loss_contrastive, edist, predictions = criterion(output1,output2,Y)
#edist = F.pairwise_distance(output1, output2, keepdim = True)
```


```python
for i in range(10):
    z = torch.cat((X0[i:i+1],X1[i:i+1]),0).cpu()
    d = edist[i].cpu().item()
    pred = int(predictions[i].cpu().item())
    imshow(torchvision.utils.make_grid(z),'Dissimilarity: {:.2f}   {}'.format(d,pred))
```


    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_0.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_1.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_2.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_3.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_4.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_5.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_6.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_7.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_8.png "xxx"){:height="100%" width="100%"}    
![an image alt text]({{ site.baseurl }}/images/oooutput_20_9.png "xxx"){:height="100%" width="100%"}    



```python

```
