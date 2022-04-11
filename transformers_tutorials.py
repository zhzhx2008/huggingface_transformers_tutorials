#encoding=utf-8

# from:http://fancyerii.github.io/2021/05/11/huggingface-transformers-1/

from transformers import pipeline
import transformers
import torch
import sys
import platform

print(platform.python_version())
print(platform.python_version_tuple())
print(type(platform.python_version_tuple()))
print('python version')
print(sys.version)
print('version info')
print(sys.version_info)
print('pytorch version')
print(torch.__version__)
print('transformers version')
print(transformers.__version__)

classifier = pipeline('sentiment-analysis')
results = classifier([
    'We are very happy to show you the transformers library.',
    'We hope you do not hate it.',
])
for result in results:
    print(result)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=pt_model, tokenizer=tokenizer)
results = classifier([
    'We are very happy to show you the transformers library.',
    'We hope you do not hate it.',
])
for result in results:
    print(result)

inputs = tokenizer('We are very happy to show you the transformers library.')
print(inputs)
pt_batch = tokenizer([
    'We are very happy to show you the transformers library.',
    'We hope you do not hate it.', ],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt')
for k, v in pt_batch.items():
    print(k, v)

pt_outputs = pt_model(**pt_batch)
print(pt_outputs)
import torch.nn.functional as F
pt_predictions = F.softmax(pt_outputs[0], dim=-1)
print(pt_predictions)
import torch
pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1, 0]))
print(pt_outputs)
pt_outputs = pt_model(**pt_batch, output_hidden_states = True, output_attentions = True)
all_hidden_states, all_attentions = pt_outputs[-2:]
# print(all_hidden_states)
# print(all_attentions)

# 具体的模型类
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# 自定义模型
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4 * 512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)

# 只改变最后一层
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
model_name = 'distilbert-base-uncased'
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
sequence = 'A Titan RTX has 24GB of VRAM'
tokenizer_sequence = tokenizer.tokenize(sequence)
print(tokenizer_sequence)
inputs = tokenizer(sequence)
print(inputs)
decoded_sequence = tokenizer.decode(inputs['input_ids'])
print(decoded_sequence)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
sequence_a = 'This is a short sequence.'
sequence_b = 'This is a rather long sequence. It is at least longer than the sequence A.'
encoded_sequence_a = tokenizer(sequence_a)
encoded_sequence_b = tokenizer(sequence_b)
print('='*50)
print(encoded_sequence_a)
print(encoded_sequence_b)
print(len(encoded_sequence_a['input_ids']), len(encoded_sequence_b['input_ids']))
padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
# print(padded_sequences['input_ids'])
# print(padded_sequences['attention_mask'])
print(padded_sequences)

print('='*50)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
sequence_a = 'HuggingFace is based in NYC'
sequence_b = 'Where is HuggingFace based?'
encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict['input_ids'])
print(decoded)

# 分类
print('='*50)
from transformers import pipeline
nlp = pipeline('sentiment-analysis')
result = nlp('I hate you')
print(result)
result = nlp('I love you')
print(result)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model_name = 'bert-base-cased-finetuned-mrpc'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classes = ['not paraphrase', 'is paraphase']
sequence_0 = 'The company HuggingFace is based in New York City'
sequence_1 = 'Apples are especially bad for your health'
sequence_2 = 'HuggingFace headquaters are situated in Manhattan'
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors='pt')
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors='pt')
paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits
paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]
print(paraphrase_results)
print(not_paraphrase_results)
