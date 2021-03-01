---
layout: post
title: How is cross entropy computed in pytorch ? 
categories: ['DeepLearning','pytorch']
---

```python
import torch
from torch import nn
import numpy as np
```

### Cross entropy with 4 outputs and a target class.

logits tensor has shape `batch_size` x `num_outputs`. Here `batch_size`=1, `num_outputs`=4


```python
ce_loss = nn.CrossEntropyLoss(reduction='none')
logits = torch.from_numpy(np.array([[4, 3, 2, 10]])).float()
print('Logits=',logits.shape,'\n',logits)

target_class = 3
target_class_tensor = torch.from_numpy(np.array([target_class]))
print('\nTarget=',target_class_tensor)
loss = ce_loss(logits, target_class_tensor)
print('\nloss=',loss)
```

    Logits= torch.Size([1, 4]) 
     tensor([[ 4.,  3.,  2., 10.]])
    
    Target= tensor([3])
    
    loss= tensor([0.0037])


You can get the probabilities from logits by using softmax. 
Then negative log of the probability of the target class is the CE loss, same as above. The loss is close to 0 
as the logit corresponding to (correct) target class 3 is much higher compared to the logits of other target classes.


```python
probabilities = nn.functional.softmax(logits,dim=1)
target_class = torch.from_numpy(np.array([3]))
print('\nProbabilities=',probabilities.shape)
print(probabilities)

print('CE loss=',-np.log(torch.squeeze(probabilities)[target_class]))
```

    
    Probabilities= torch.Size([1, 4])
    tensor([[2.4696e-03, 9.0850e-04, 3.3422e-04, 9.9629e-01]])
    CE loss= tensor([0.0037])



![an image alt text]({{ site.baseurl }}/images/pytorch_crossentropy_1.jpg "xxx"){:height="40%" width="40%"} |


### Cross entropy with 4 outputs and a target class. Each output is a 2D tensor.

logits tensor has shape `batch_size` x `num_outputs` x `2`. Here batch_size=1, num_outputs=4

The loss is very high as the logit corresponding to target class 3 is low. The highest logit
corresponds to the target class 0.



```python
ce_loss = nn.CrossEntropyLoss(reduction='none')
logits = torch.from_numpy(np.array([[20, 2, 3, 4]])).float()
print('Logits',logits.shape,'\n',logits)
probabilities = nn.functional.softmax(logits,dim=1)
target_classes = torch.from_numpy(np.array([3]))
print('\nProbabilities=',probabilities.shape)
print(probabilities)
print('Target',target_classes.shape,'\n',target_classes)
loss = ce_loss(logits, target_classes)
print('\nloss=',loss)
```

    Logits torch.Size([1, 4]) 
     tensor([[20.,  2.,  3.,  4.]])
    
    Probabilities= torch.Size([1, 4])
    tensor([[1.0000e+00, 1.5230e-08, 4.1399e-08, 1.1254e-07]])
    Target torch.Size([1]) 
     tensor([3])
    
    loss= tensor([16.])


### Cross entropy with 4 outputs and a target class. Each output is a 2D tensor.

logits tensor has shape `batch_size` x `num_outputs` x `2`. Here batch_size=1, num_outputs=4. 
The loss tensor will have dimensionality of 2 - one loss for each of the 2 dimensions of the output (as we have reduction=`none`)

ce_loss = nn.CrossEntropyLoss(reduction='none')

logits = torch.from_numpy(np.array([[[4,10], [3,2], [2,3], [10,4]]])).float()
print('Logits',logits.shape,'\n',logits)

#Target classes for each dimension of the output.
target_classes = torch.from_numpy(np.array([[3,1]]))
print('\nTarget',target_classes.shape,'\n',target_classes)

loss = ce_loss(logits, target_classes)
print('\nloss=',loss)

![an image alt text]({{ site.baseurl }}/images/pytorch_crossentropy_2.jpg "xxx"){:height="40%" width="40%"} |

### Cross entropy with 4 outputs and a target class (batch_size=2). Each output is a 2D tensor.

logits tensor has shape `batch_size` x `num_outputs` x `2`. Here batch_size=2, num_outputs=4. 
The loss tensor will have dimensionality of 2 - one loss for each of the 2 dimensions of the output 
(as we have reduction=`none`). This is repeated for each of the 2 elements of the batch.


```python
ce_loss = nn.CrossEntropyLoss(reduction='none')

logits = torch.from_numpy(np.array([[[4,4], [3,3], [2,2], [2,20]], [[20,20], [3,3], [2,2], [10,10]]])).float()

print('Logits',logits.shape,'\n',logits)

target_classes = torch.from_numpy(np.array([[3,3],[0,0]]))

print(target_classes.shape,'\n',target_classes)

loss = ce_loss(logits, target_classes)
print('\nloss=',loss)
```

    Logits torch.Size([2, 4, 2]) 
     tensor([[[ 4.,  4.],
             [ 3.,  3.],
             [ 2.,  2.],
             [ 2., 20.]],
    
            [[20., 20.],
             [ 3.,  3.],
             [ 2.,  2.],
             [10., 10.]]])
    torch.Size([2, 2]) 
     tensor([[3, 3],
            [0, 0]])
    
    loss= tensor([[2.4938e+00, 1.1921e-07],
            [4.5418e-05, 4.5418e-05]])

