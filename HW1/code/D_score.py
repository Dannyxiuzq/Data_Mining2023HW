import numpy as np
import spacy
import pandas as pd
import requests
from PIL.Image import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black', 'white']
num_list = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
            'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
            'million': 1000000}
processor = ViltProcessor.from_pretrained("model/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("model/vilt-b32-finetuned-vqa")
QA = pipeline('question-answering', model="model/roberta-base-squad2", tokenizer="model/roberta-base-squad2")
object_template = "Can you find any %s?"
color_template = "Is the %s %s?"
num_template = "%s is the number of %s?"
relation_q_template = "Is %s?"
relation_template = "What is the relation of '%s' and '%s'?"
self_template = "What is the %s doing?"


def object_extraction(objects):
    objects_dict = {}
    new_objects = []
    for chunk in objects:
        words = chunk.split(' ')
        num = None
        color = None
        flag = 0
        for i in range(len(words)):
            if words[i] in num_list.keys():
                num = num_list[words[i]]
            elif words[i].isdigit():
                num = int(words[i])
            elif words[i] in color_list:
                color = words[i]
                flag = 1
                break
        object = ' '.join(words[i + flag:])
        new_objects.append(object)
        objects_dict[object] = {}
        objects_dict[object]['Num'] = num
        objects_dict[object]['Color'] = color
    return objects_dict, new_objects


def vqa(image, question, target):
    encoding = processor(image, question, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits.softmax(dim=-1).detach().numpy()
    idx = model.config.label2id[target]
    return logits[0][idx]


def tqa(question, context):
    QA_input = {
        'question': question,
        'context': context
    }
    res = QA(QA_input)
    return res


def get_matrix(prompt):
    doc = nlp(prompt)
    objects = [chunk.text for chunk in doc.noun_chunks if chunk.text not in stop_words]
    objects_dict, objects = object_extraction(objects)
    columns = ['Num', 'Color', 'Existence']
    columns.extend(objects)
    descriptive_matrix = pd.DataFrame.from_dict(objects_dict, orient='index')
    score_matrix = pd.DataFrame(columns=columns, index=objects)
    descriptive_matrix['Existence'] = None
    for o in objects:
        descriptive_matrix[o] = None
    # print(descriptive_matrix)
    for i in range(len(objects)):
        descriptive_matrix.loc[objects[i], objects[i]] = tqa(self_template % objects[i], prompt)['answer']
        for j in range(i + 1, len(objects)):
            descriptive_matrix.loc[objects[i], objects[j]] = tqa(relation_template % (objects[i], objects[j]), prompt)[
                'answer']
    return descriptive_matrix, score_matrix



def get_score(descriptive_matrix, score_matrix, image=None):
    for i in descriptive_matrix.columns:
        for j in descriptive_matrix.index:
            if i == 'Existence':
                question = object_template % j
                score_matrix.loc[j, i] = vqa(image, question, 'yes')
            elif i == 'Color' and descriptive_matrix.loc[j, i] is not None:
                question = color_template % (j, descriptive_matrix.loc[j, i])
                score_matrix.loc[j, i] = vqa(image, question, 'yes')
            elif i == 'Num' and descriptive_matrix.loc[j, i] is not None:
                #question = num_template % (descriptive_matrix.loc[j, i], j)
                #score_matrix.loc[j, i] = vqa(image, question, 'yes')
                pass
            elif descriptive_matrix.loc[j, i] is not None:
                question = relation_q_template % descriptive_matrix.loc[j, i]
                score_matrix.loc[j, i] = vqa(image, question, 'yes')
    return score_matrix


def get_D_score(img, prompt):
    descriptive_matrix, score_matrix = get_matrix(prompt)
    score_matrix = get_score(descriptive_matrix, score_matrix, img)
    score_matrix = score_matrix.values
    describe_score = score_matrix.mean(axis=None)
    #print(describe_score)
    if np.isnan(describe_score):
        describe_score = 0
    return describe_score


def get_batch_score(batch_path, batch_prompt):
    batch_describe_score = []
    for i in range(len(batch_prompt)):
        img_path = batch_path[i]
        prompt = batch_prompt[i]
        img = Image.open(img_path)
        #descriptive_matrix, score_matrix = get_matrix(prompt)
        describe_score = get_D_score(img, prompt)
        batch_describe_score.append(describe_score)
    return batch_describe_score


'''
img_path = "Project_Dataset/Selected_Train_Dataset/a yellow t-shirt with a dog on it_/bad/a-yellow-t-shirt-with-a-dog-on-3.png"
img = Image.open(img_path)
text = "a yellow t-shirt with a dog on it"
#text = "beautiful fireworks in the sky with red, white and blue"
#text = "Face of an orange frog in cartoon style"
#text = "the words 'KEEP OFF THE GRASS'"
#text = "a dream"
descriptive_matrix, score_matrix = get_matrix(text)
score_matrix = get_score(descriptive_matrix, score_matrix, img)
print(score_matrix.mean(axis=None))
#print(descriptive_matrix)
#print(score_matrix)
#print(get_score(descriptive_matrix, score_matrix, img))
#print(vqa(img, 'Can you find any elephant?', 'yes'))
'''

