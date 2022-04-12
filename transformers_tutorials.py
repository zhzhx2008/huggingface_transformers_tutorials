# encoding=utf-8

# from:http://fancyerii.github.io/2021/05/11/huggingface-transformers-1/

import platform
import sys

import torch
import transformers
from transformers import pipeline

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


# 3.7.4
# ('3', '7', '4')
# <class 'tuple'>
# python version
# 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
# version info
# sys.version_info(major=3, minor=7, micro=4, releaselevel='final', serial=0)
# pytorch version
# 1.11.0+cpu
# transformers version
# 4.2.1

def sentiment_analysis():
    global result, model_name, tokenizer, inputs
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
    pt_outputs = pt_model(**pt_batch, labels=torch.tensor([1, 0]))
    print(pt_outputs)
    pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
    all_hidden_states, all_attentions = pt_outputs[-2:]
    # print(all_hidden_states)
    # print(all_attentions)


def self_define_model():
    global model_name, model, tokenizer
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
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    model_name = 'distilbert-base-uncased'
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)


def tokenizer_a():
    global tokenizer, inputs
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
    print('=' * 50)
    print(encoded_sequence_a)
    print(encoded_sequence_b)
    print(len(encoded_sequence_a['input_ids']), len(encoded_sequence_b['input_ids']))
    padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
    # print(padded_sequences['input_ids'])
    # print(padded_sequences['attention_mask'])
    print(padded_sequences)
    print('=' * 50)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    sequence_a = 'HuggingFace is based in NYC'
    sequence_b = 'Where is HuggingFace based?'
    encoded_dict = tokenizer(sequence_a, sequence_b)
    decoded = tokenizer.decode(encoded_dict['input_ids'])
    print(decoded)


def classify():
    global result, model_name, tokenizer, model
    # 分类
    print('=' * 50)
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
    print('paraphrase={}'.format(paraphrase))
    print('not_paraphrase={}'.format(not_paraphrase))
    paraphrase_classification = model(**paraphrase)
    not_paraphrase_classification = model(**not_paraphrase)
    paraphrase_classification_logits = paraphrase_classification.logits
    not_paraphrase_classification_logits = not_paraphrase_classification.logits
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]
    print('paraphrase_results={}'.format(paraphrase_results))
    print('not_paraphrase_results={}'.format(not_paraphrase_results))


def qa():
    # 抽取式问答
    # from transformers import pipeline
    # nlp = pipeline('question-answering')    # RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
    # context = r"""
    # Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
    # question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
    # a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
    # """
    # result = nlp(question='What is extractive question answering?', context=context)
    # print('result={}'.format(result))
    # result = nlp(question='What is a good example of a question answering dataset?', context=context)
    # print('result={}'.format(result))

    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    text = r"""
    Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
    """
    questions = [
        "How many pretrained models are available in 🤗 Transformers?",
        "What does 🤗 Transformers provide?",
        "🤗 Transformers provides interoperability between which frameworks?",
    ]
    for question in questions:
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt')
        input_ids = inputs['input_ids'].tolist()[0]
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start_scores = torch.argmax(
            answer_start_scores
        )
        answer_end_scores = torch.argmax(
            answer_end_scores
        ) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start_scores:answer_end_scores]))
        print('q:{}'.format(question))
        print('a:{}'.format(answer))


def mlm():
    from transformers import pipeline
    nlp = pipeline('fill-mask')
    from pprint import pprint
    pprint(nlp(f'HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks.'))

    from transformers import AutoModelWithLMHead, AutoTokenizer
    import torch
    model_name = 'distilbert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    print(tokenizer.mask_token)
    sequence = f'Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint.'
    input = tokenizer.encode(sequence, return_tensors='pt')
    mask_token_logits = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input).logits
    mask_token_logits = token_logits[0, mask_token_logits, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    for token in top_5_tokens:
        print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))


def clm():
    from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
    import torch
    from torch.nn import functional as F
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    sequence = f'Hugging Face is based in DUMBO, New York City, and'
    input_ids = tokenizer.encode(sequence, return_tensors='pt')
    next_token_logits = model(input_ids).logits[:, -1, :]
    filter_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    probs = F.softmax(filter_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat([input_ids, next_token], dim=-1)
    resulting_string = tokenizer.decode(generated.tolist()[0])
    print(resulting_string)


def text_generation():
    from transformers import pipeline
    text_generator = pipeline('text-generation')
    text_generator = pipeline('text-generation', model='xlnet-base-cased')
    print(text_generator('As far as I am concerned, I will', max_length=50, do_sample=False))


def ner():
    from transformers import pipeline
    nlp = pipeline('ner')
    sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is visible from the window."
    print(nlp(sequence))


def abstract():
    from transformers import pipeline
    summarizer = pipeline("summarization")
    ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """
    print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

    from transformers import AutoModelWithLMHead, AutoTokenizer
    model_name = 't5-base'
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode('summarize: ' + ARTICLE, return_tensors='pt', max_length=512)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    print(outputs)


def translation():
    from transformers import pipeline
    translator = pipeline('translation_en_to_de')
    print(
        translator('Hugging Face is a technology company based in New York and Paris', max_length=40)
    )

    from transformers import AutoModelWithLMHead, AutoTokenizer
    model_name = 't5-base'
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode('translate English to German: Hugging Face is a technology company based in New York and Paris', return_tensors='pt')
    outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
    print(outputs)
    resulting_string = tokenizer.decode(outputs.tolist()[0])
    print(resulting_string)

    from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
    model_name = 'Helsinki-NLP/opus-mt-en-zh'
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translator = pipeline('translation_en_to_zh', model=model, tokenizer=tokenizer)
    text = 'I like to study Data Science and Machine Learning'
    translated_text = translator(text, max_length=40)[0]['translation_text']
    print(translated_text)


def tokenizer_data():
    # AutoTokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    encoded_input = tokenizer("Hello, I'm a single sentence!")
    print(encoded_input)
    print(tokenizer.decode(encoded_input['input_ids']))
    print(tokenizer.decode(encoded_input['input_ids'], skip_special_tokens=True))
    batch_sentences = ["Hello I'm a single sentence",
                       "And another sentence",
                       "And the very very last one"]
    encoded_input = tokenizer(batch_sentences)
    print(encoded_input)
    print(tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt'))

    # two inputs
    encoded_input = tokenizer("How old are you?", "I'm 6 years old")
    print(encoded_input)
    print(tokenizer.decode(encoded_input['input_ids']))
    batch_sentences = ["Hello I'm a single sentence",
                       "And another sentence",
                       "And the very very last one"]
    batch_of_second_sentences = ["I'm a sentence that goes with the first sentence",
                                 "And I should be encoded with the second sentence",
                                 "And I go with the very last one"]
    encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences)
    print(encoded_inputs)
    for ids in encoded_inputs['input_ids']:
        print(tokenizer.decode(ids))
    print(tokenizer(batch_sentences, batch_of_second_sentences, padding=True, truncation=True, return_tensors='pt'))

    encoded_input = tokenizer(["Hello", "I'm", "a", "single", "sentence"], is_split_into_words=True)
    print(encoded_input)

    batch_sentences = [["Hello", "I'm", "a", "single", "sentence"],
                   ["And", "another", "sentence"],
                   ["And", "the", "very", "very", "last", "one"]]
    encoded_input = tokenizer(batch_sentences, is_split_into_words=True)
    print(encoded_input)
    batch_of_second_sentences = [["I'm", "a", "sentence", "that", "goes", "with", "the", "first", "sentence"],
                                 ["And", "I", "should", "be", "encoded", "with", "the", "second", "sentence"],
                                 ["And", "I", "go", "with", "the", "very", "last", "one"]]
    encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences, is_split_into_words=True)
    print(encoded_inputs)
    batch = tokenizer(batch_sentences, batch_of_second_sentences, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
    print(batch)


def fine_tuning_pt():
    from transformers import BertForSequenceClassification, AdamW, BertTokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.train()
    no_decay = ['bias', 'LyaerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_batch = ["I love Pixar.", "I don't care for Pixar."]
    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    labels = torch.tensor([1, 0]).unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # 自己计算loss
    from torch.nn import functional as F
    labels = torch.tensor([1, 0])
    outputs = model(input_ids, attention_mask=attention_mask)
    loss = F.cross_entropy(outputs.logits, labels)
    loss.backward()
    optimizer.step()

    # from transformers import get_linear_schedule_with_warmup
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=2)
    # loss.backward()
    # optimizer.step()
    # scheduler.step()

    # freeze某些参数
    for param in model.base_model.parameters():
        param.requires_grad = False


def fine_tuning_tf():
    # from transformers import TFBertForSequenceClassification
    # model_name = 'bert-base-uncased'
    # model = TFBertForSequenceClassification.from_pretrained(model_name)
    # from transformers import BertTokenizer, glue_convert_examples_to_features
    # import tensorflow as tf
    # import tensorflow_datasets as tfds
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # data = tfs.load('glue/mrpc')
    # train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
    # train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # model.compile(optimizer=optimizer, loss=loss)
    # model.fit(train_dataset, epochs=2, steps_per_epoch=115)
    # from transformers import BertForSequenceClassification
    # model.save_pretrained('./')
    # pytorch_model = BertForSequenceClassification.from_pretrained('./', from_tf=True)
    pass


def fine_tuning_trainer():
    from transformers import BertForSequenceClassification, Trainer, TrainingArguments
    model_name = 'bert-large-uncased'
    model = BertForSequenceClassification.from_pretrained(model_name)
    trainning_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    train_dataset, test_dataset = None, None
    trainner = Trainer(
        model=model,
        args=trainning_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


if __name__ == '__main__':
    # sentiment_analysis()
    # self_define_model()
    # tokenizer_a()
    # classify()
    # qa()
    # mlm()
    # clm()
    # text_generation()
    # ner()
    # abstract()
    # translation()
    # tokenizer_data()
    fine_tuning_pt()
    pass
