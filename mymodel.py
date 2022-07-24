import torch.nn as nn
import transformers

class BertClassificationModel(nn.Module):
    def __init__(self, ptm_path="bert-base-chinese/", hidden_size=768, class_num=3, dropout=0):
        super(BertClassificationModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(ptm_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        #self.classifier = nn.Linear(768, 3)  # hidden_size=768，分三类
        self.classifier = nn.Linear(hidden_size, class_num)  # hidden_size=768，分三类
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # 获得预训练模型的输出
        bert_cls_hidden_state = bert_output[1]
        bert_cls_hidden_state = self.dropout(bert_cls_hidden_state)
        logits = self.classifier(bert_cls_hidden_state)
        return logits
