---
layout: post
title: Using Roberta classification head for fine-tuning a pre-trained BERT model 
categories: ['DeepLearning','Pytorch','NLP', 'Huggingface', 'BERT']
---


```python
import os
import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import (
    Dataset, 
    DataLoader, 
    RandomSampler, 
    SequentialSampler
)

import math 
from transformers import  (
    BertPreTrainedModel, 
    RobertaConfig, 
    RobertaTokenizerFast
)

from transformers.optimization import (
    AdamW, 
    get_linear_schedule_with_warmup
)

from scipy.special import softmax
from torch.nn import CrossEntropyLoss

from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc,
    average_precision_score,
)

from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)
```

## Load some training and testing data for classification task


```python
url = 'https://raw.githubusercontent.com/pchanda/pchanda.github.io/master/data/bertMol/train_data.txt'
train_df = pd.read_csv(url,delimiter='\t')
print('Training data...')
print (train_df.head())

url = 'https://raw.githubusercontent.com/pchanda/pchanda.github.io/master/data/bertMol/test_data.txt'
test_df = pd.read_csv(url,delimiter='\t')
print('\nTesting data...')
print (test_df.head())
```

    Training data...
                                                    text  label
    0                 COc1cc(nc(OC)n1)c2cn(nc2C)c3cccnc3      0
    1     CCN(C(=O)NC(=O)c1ccc(F)cc1)c2cn(nc2Cl)c3cccnc3      0
    2             CCc1noc(n1)C2CCN(CC2)C(=O)N(C)c3cccnc3      0
    3               CCN(C(=O)C(C)CBr)c1sc(nc1Cl)c2cccnc2      0
    4  CCS(=NC(=O)c1cc(Cl)cc(C)c1NC(=O)c2cnc(nc2c3ncc...      0
    
    Testing data...
                                                    text  label
    0             CN(C(=O)CCC(F)(F)F)c1cn(nc1Br)c2cccnc2      0
    1  COC1CCC2(CC1)NC(=O)C(=C2[O-])c3c(C)cc(cc3C)n4c...      0
    2        CSCCN(C(=O)OC(C)(C)C)c1sc([nH+]c1C)c2cccnc2      0
    3                     CCOC(=O)c1cn2nc(sc2n1)c3cccnc3      0
    4                      CSCCC(=O)Nc1sc(nc1Cl)c2cccnc2      0


## Define parameters for the fine-tuning task


```python
model_name = 'pchanda/pretrained-smiles-pubchem10m'
num_labels = 2
device = torch.device("cuda")

tokenizer_name = model_name

max_seq_length = 128 
train_batch_size = 8
test_batch_size = 8
warmup_ratio = 0.06
weight_decay=0.0
gradient_accumulation_steps = 1
num_train_epochs = 25
learning_rate = 1e-05
adam_epsilon = 1e-08
```

## Define a classification head based on Roberta


```python
class RobertaForSmilesClassification(BertPreTrainedModel):
    
    def __init__(self, config):
        super(RobertaForSmilesClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        
        
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.roberta(input_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

```

## Load pre-trained Roberta model and tokenizer


```python
config_class = RobertaConfig
model_class = RobertaForSmilesClassification
tokenizer_class = RobertaTokenizerFast

config = config_class.from_pretrained(model_name, num_labels=num_labels)

model = model_class.from_pretrained(model_name, config=config)
print('Model=\n',model,'\n')

tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
print('Tokenizer=',tokenizer,'\n')
```

    Some weights of the model checkpoint at pchanda/pretrained-smiles-pubchem10m were not used when initializing RobertaForSmilesClassification: ['lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.decoder.bias']
    - This IS expected if you are initializing RobertaForSmilesClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSmilesClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of RobertaForSmilesClassification were not initialized from the model checkpoint at pchanda/pretrained-smiles-pubchem10m and are newly initialized: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias', 'classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


    Model=
     RobertaForSmilesClassification(
      (roberta): RobertaModel(
        (embeddings): RobertaEmbeddings(
          (word_embeddings): Embedding(52000, 768, padding_idx=1)
          (position_embeddings): Embedding(514, 768, padding_idx=1)
          (token_type_embeddings): Embedding(1, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): RobertaEncoder(
          (layer): ModuleList(
            (0): RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1): RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (2): RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (3): RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (4): RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (5): RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (pooler): RobertaPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (classifier): RobertaClassificationHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (out_proj): Linear(in_features=768, out_features=2, bias=True)
      )
    ) 
    


    Special tokens have been added in the vocabulary, make sure the associated word embedding are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embedding are fine-tuned or trained.


    Tokenizer= PreTrainedTokenizerFast(name_or_path='pchanda/pretrained-smiles-pubchem10m', vocab_size=591, model_max_len=514, is_fast=True, padding_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': AddedToken("[MASK]", rstrip=False, lstrip=True, single_word=False, normalized=False)}) 
    


## Define custom class to convert text and labels into a Dataset object with encoded text and labels


```python
class MyClassificationDataset(Dataset):
    
    def __init__(self, data, tokenizer):
        text, labels = data
        self.examples = tokenizer(text=text,text_pair=None,truncation=True,padding="max_length",
                                  max_length=max_seq_length,return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
        

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.labels[index]


train_examples = (train_df.iloc[:, 0].astype(str).tolist(), train_df.iloc[:, 1].tolist())
train_dataset = MyClassificationDataset(train_examples,tokenizer)

test_examples = (test_df.iloc[:, 0].astype(str).tolist(), test_df.iloc[:, 1].tolist())
test_dataset = MyClassificationDataset(test_examples,tokenizer)
```

### Methods to prepare a batch from train (and test) datasets


```python
def get_inputs_dict(batch):
    inputs = {key: value.squeeze(1).to(device) for key, value in batch[0].items()}
    inputs["labels"] = batch[1].to(device)
    return inputs

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,sampler=train_sampler,batch_size=train_batch_size)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

#Extract a batch as sanity-check
batch = get_inputs_dict(next(iter(train_dataloader)))
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)
labels = batch['labels'].to(device)

print(batch)
```

    {'input_ids': tensor([[12, 16, 16,  ...,  0,  0,  0],
            [12, 16, 34,  ...,  0,  0,  0],
            [12, 16, 16,  ...,  0,  0,  0],
            ...,
            [12, 16, 16,  ...,  0,  0,  0],
            [12, 16, 23,  ...,  0,  0,  0],
            [12, 16, 16,  ...,  0,  0,  0]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            ...,
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0'), 'labels': tensor([0, 0, 0, 1, 0, 0, 0, 0], device='cuda:0')}


## Define parameters for optimizer and scheduler


```python
t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
optimizer_grouped_parameters = []
custom_parameter_names = set()
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters.extend(
    [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in custom_parameter_names and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
)

warmup_steps = math.ceil(t_total * warmup_ratio)
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
```

## Method to compute accuracy of predictions


```python
def compute_metrics(preds, model_outputs, labels, eval_examples=None, multi_label=False):
    assert len(preds) == len(labels)
    mismatched = labels != preds
    wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    scores = np.array([softmax(element)[1] for element in model_outputs])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    auprc = average_precision_score(labels, scores)
    return (
        {
            **{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "auroc": auroc, "auprc": auprc},
        },
        wrong,
    )

def print_confusion_matrix(result):
    print('confusion matrix:')
    print('            predicted    ')
    print('          0     |     1')
    print('    ----------------------')
    print('   0 | ',format(result['tn'],'5d'),' | ',format(result['fp'],'5d'))
    print('gt -----------------------')
    print('   1 | ',format(result['fn'],'5d'),' | ',format(result['tp'],'5d'))
    print('---------------------------------------------------')
```

## Training and Evaluation


```python
model.to(device)

model.zero_grad()

for epoch in range(num_train_epochs):

    model.train()
    epoch_loss = []
    
    for batch in train_dataloader:
        batch = get_inputs_dict(batch)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        epoch_loss.append(loss.item())
        
    #evaluate model with test_df at the end of the epoch.
    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(test_dataloader)
    preds = np.empty((len(test_dataset), num_labels))
    out_label_ids = np.empty((len(test_dataset)))
    model.eval()
    
    for i,test_batch in enumerate(test_dataloader):
        with torch.no_grad():
            test_batch = get_inputs_dict(test_batch)
            input_ids = test_batch['input_ids'].to(device)
            attention_mask = test_batch['attention_mask'].to(device)
            labels = test_batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()
            
        nb_eval_steps += 1
        start_index = test_batch_size * i
        end_index = start_index + test_batch_size if i != (n_batches - 1) else len(test_dataset)
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = test_batch["labels"].detach().cpu().numpy()
        
    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds
    preds = np.argmax(preds, axis=1)
    result, wrong = compute_metrics(preds, model_outputs, out_label_ids, test_examples)
    
    print('epoch',epoch,'Training avg loss',np.mean(epoch_loss))
    print('epoch',epoch,'Testing  avg loss',eval_loss)
    print(result)
    print_confusion_matrix(result)
    print('---------------------------------------------------\n')
```

    epoch 0 Training avg loss 0.4784002370580479
    epoch 0 Testing  avg loss 0.39766009540661523
    {'mcc': 0.23128495510646516, 'tp': 47, 'tn': 1463, 'fp': 28, 'fn': 299, 'auroc': 0.790707636958553, 'auprc': 0.4869804828396154}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1463  |     28
    gt -----------------------
       1 |    299  |     47
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 1 Training avg loss 0.39961886811587544
    epoch 1 Testing  avg loss 0.49644354727605117
    {'mcc': 0.39038012336992456, 'tp': 252, 'tn': 1112, 'fp': 379, 'fn': 94, 'auroc': 0.8101809314460947, 'auprc': 0.5245091787941775}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1112  |    379
    gt -----------------------
       1 |     94  |    252
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 2 Training avg loss 0.3579287456452019
    epoch 2 Testing  avg loss 0.36474067498322416
    {'mcc': 0.44789647834407514, 'tp': 160, 'tn': 1394, 'fp': 97, 'fn': 186, 'auroc': 0.8353318368786904, 'auprc': 0.5621066375112733}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1394  |     97
    gt -----------------------
       1 |    186  |    160
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 3 Training avg loss 0.31056923224349264
    epoch 3 Testing  avg loss 0.4161553577884384
    {'mcc': 0.42401967141302777, 'tp': 198, 'tn': 1299, 'fp': 192, 'fn': 148, 'auroc': 0.8218908828694711, 'auprc': 0.5376534760438139}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1299  |    192
    gt -----------------------
       1 |    148  |    198
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 4 Training avg loss 0.2766777211162089
    epoch 4 Testing  avg loss 0.3893753176797991
    {'mcc': 0.47703926582985007, 'tp': 193, 'tn': 1356, 'fp': 135, 'fn': 153, 'auroc': 0.8337462152491053, 'auprc': 0.5773530213284616}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1356  |    135
    gt -----------------------
       1 |    153  |    193
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 5 Training avg loss 0.24243826554932943
    epoch 5 Testing  avg loss 0.437101148965809
    {'mcc': 0.4553555157636891, 'tp': 205, 'tn': 1313, 'fp': 178, 'fn': 141, 'auroc': 0.839904552556185, 'auprc': 0.5790069624890034}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1313  |    178
    gt -----------------------
       1 |    141  |    205
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 6 Training avg loss 0.2114253447335041
    epoch 6 Testing  avg loss 0.43283067895909366
    {'mcc': 0.4803595327768338, 'tp': 211, 'tn': 1323, 'fp': 168, 'fn': 135, 'auroc': 0.8449735018977061, 'auprc': 0.5913977265027471}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1323  |    168
    gt -----------------------
       1 |    135  |    211
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 7 Training avg loss 0.17955164654197656
    epoch 7 Testing  avg loss 0.4619454271377712
    {'mcc': 0.4484367040366249, 'tp': 171, 'tn': 1375, 'fp': 116, 'fn': 175, 'auroc': 0.843083549466355, 'auprc': 0.5948429377070619}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1375  |    116
    gt -----------------------
       1 |    175  |    171
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 8 Training avg loss 0.16105714170370009
    epoch 8 Testing  avg loss 0.5430390530296237
    {'mcc': 0.46873777704605685, 'tp': 189, 'tn': 1357, 'fp': 134, 'fn': 157, 'auroc': 0.8349790457581713, 'auprc': 0.5863953064904349}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1357  |    134
    gt -----------------------
       1 |    157  |    189
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 9 Training avg loss 0.13479235491260488
    epoch 9 Testing  avg loss 0.6013538978383978
    {'mcc': 0.4729228194676383, 'tp': 213, 'tn': 1312, 'fp': 179, 'fn': 133, 'auroc': 0.8431048720066061, 'auprc': 0.5893703188904773}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1312  |    179
    gt -----------------------
       1 |    133  |    213
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 10 Training avg loss 0.11649735385942256
    epoch 10 Testing  avg loss 0.656377293797635
    {'mcc': 0.4399251717391603, 'tp': 210, 'tn': 1287, 'fp': 204, 'fn': 136, 'auroc': 0.8262251737787031, 'auprc': 0.5638031705362816}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1287  |    204
    gt -----------------------
       1 |    136  |    210
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 11 Training avg loss 0.09920337930326005
    epoch 11 Testing  avg loss 0.6670230242555313
    {'mcc': 0.46635941932306035, 'tp': 205, 'tn': 1323, 'fp': 168, 'fn': 141, 'auroc': 0.8309452088252056, 'auprc': 0.5807697397848588}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1323  |    168
    gt -----------------------
       1 |    141  |    205
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 12 Training avg loss 0.08404365494993894
    epoch 12 Testing  avg loss 0.7289287104663621
    {'mcc': 0.472015777555128, 'tp': 214, 'tn': 1309, 'fp': 182, 'fn': 132, 'auroc': 0.8368883823170234, 'auprc': 0.5848170685521272}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1309  |    182
    gt -----------------------
       1 |    132  |    214
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 13 Training avg loss 0.07693617246798197
    epoch 13 Testing  avg loss 0.7343728942086708
    {'mcc': 0.46251662354649953, 'tp': 200, 'tn': 1330, 'fp': 161, 'fn': 146, 'auroc': 0.831829125039253, 'auprc': 0.585384950200475}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1330  |    161
    gt -----------------------
       1 |    146  |    200
    ---------------------------------------------------
    ---------------------------------------------------
    
    epoch 14 Training avg loss 0.07127877530723775
    epoch 14 Testing  avg loss 0.7521376626402818
    {'mcc': 0.4645457723684743, 'tp': 208, 'tn': 1315, 'fp': 176, 'fn': 138, 'auroc': 0.8334050546050871, 'auprc': 0.5805451960298494}
    confusion matrix:
                predicted    
              0     |     1
        ----------------------
       0 |   1315  |    176
    gt -----------------------
       1 |    138  |    208
    ---------------------------------------------------
    ---------------------------------------------------
    

