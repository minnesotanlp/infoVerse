import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

def load_backbone(name, output_attentions=False):
    if name == 'bert':
        from transformers import BertForMaskedLM, BertTokenizer
        backbone = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=output_attentions)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.name = 'bert-base-uncased'
    elif name == 'roberta':
        from transformers import RobertaForMaskedLM, RobertaTokenizer
        backbone = RobertaForMaskedLM.from_pretrained('roberta-base', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.name = 'roberta-base'
    elif name == 'roberta_large':
        from transformers import RobertaForMaskedLM, RobertaTokenizer
        backbone = RobertaForMaskedLM.from_pretrained('roberta-large', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        tokenizer.name = 'roberta-large'
    elif name == 'roberta_mc_large':
        from transformers import RobertaForMultipleChoice, RobertaTokenizer
        backbone = RobertaForMultipleChoice.from_pretrained('roberta-large', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        tokenizer.name = 'roberta-large'    
    elif name == 'sentence_bert':
        from transformers import AutoTokenizer, AutoModel
        backbone = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        tokenizer.name = 'paraphrase-MiniLM-L6-v2'
    else:
        raise ValueError('No matching backbone network')

    return backbone, tokenizer

class Classifier(nn.Module):
    def __init__(self, backbone_name, backbone, n_classes, train_type='None'):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.dropout = nn.Dropout(0.1)
        self.n_classes = n_classes
        self.train_type = train_type

        if 'large' in backbone_name:
            n_dim = 1024
        elif 'sent' in backbone_name:
            n_dim = 384
        else:
            n_dim = 768
        ##### Classifier for down-stream task
        self.net_cls = nn.Linear(n_dim, n_classes)

    def forward(self, x, inputs_embed=None, get_penul=False, lm=False, sent=False):
        if self.backbone_name in ['bert', 'albert']:
            attention_mask = (x > 0).float()
        else:
            attention_mask = (x != 1).float()

        if inputs_embed is not None:
            out_cls_orig = self.backbone(None, attention_mask=attention_mask, inputs_embeds=inputs_embed)[1]
        elif lm:
            lm_output = self.backbone(x, attention_mask, inputs_embeds=inputs_embed)[0]
            return lm_output
        elif sent:
            out_cls_orig = self.backbone(x, attention_mask, inputs_embeds=inputs_embed)[1]
        else:
            out_cls_orig = self.backbone(x, attention_mask=attention_mask, inputs_embeds=inputs_embed)[1]

        out_cls = self.dropout(out_cls_orig)
        out_cls = self.net_cls(out_cls)
        if self.n_classes == 1:
            out_cls = out_cls.view(-1, 2)
            
        if get_penul:
            return out_cls, out_cls_orig
        else:
            return out_cls