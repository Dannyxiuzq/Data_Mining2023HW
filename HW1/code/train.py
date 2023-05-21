import os
import glob
import clip
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

"""加载clip模型"""
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 5e-5
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)

"""读取数据集路径"""
train_data_root = r'Project_Dataset/Selected_Train_Dataset/'
path_list = os.listdir(train_data_root)
# all_good_images_path = []
# all_bad_images_path = []
all_image_path_pairs = []
for path in path_list:
    good_img_path = glob.glob(train_data_root + path + '/good/' + '*.png')
    bad_img_path = glob.glob(train_data_root + path + '/bad/' + '*.png')
    pair_img_path = list(zip(good_img_path, bad_img_path))
    # all_good_images_path.extend(good_img_path)
    # all_bad_images_path.extend(bad_img_path)
    all_image_path_pairs.extend(pair_img_path)
# print(all_image_path_pairs[0])
def get_txt_path(img_path):
    i = 0
    txt_path = ""
    for s in img_path:
        txt_path += s
        if s == '/':
            i += 1
        if i == 3:
            break
    return txt_path

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

# class train_data(Dataset):
#     def __init__(self, lable, img_path_list):
#         self.path = img_path_list
#         self.label = lable

#     def __len__(self):
#         return len(self.path)

#     def __getitem__(self, item):
#         """
#         :param item:
#         :return:
#             img:图片
#             txt:图片质量+图片描述
#             label:图片质量-good/bad 图片好/坏
#             prompt:图片描述-prompt.txt文本内容
#         """
#         img = Image.open(self.path[item])
#         img = preprocess(img)
#         txt = get_txt_path(self.path[item]) + 'prompt.txt'
#         file = open(txt, encoding='utf-8')
#         prompt = file.read(300)

#         if self.label == "good":
#             txt = "a beautiful and concise photo of " + prompt
#         else:
#             txt = "a mussy photo of " + prompt

#         # txt = "a " + self.label + " photo of " + prompt
#         return img, txt, self.label, prompt

class train_data(Dataset):
    def __init__(self, train_data_root):
        path_list = os.listdir(train_data_root)
        self.prompts = []
        # self.good_imgs = []
        # self.bad_imgs = []
        self.good_imgs_path = []
        self.bad_imgs_path = []
        self.raw_prompts = []
        for path in path_list:
            prompt_path = os.path.join(train_data_root, path, 'prompt.txt')
            raw_prompt = open(prompt_path, encoding='utf-8').read(300)
            prompt = "A photo of " + raw_prompt + '.'

            good_img_paths = glob.glob(train_data_root + path + '/good/' + '*.png')
            self.good_imgs_path.extend(good_img_paths)
            # for good_img_path in good_img_paths:
            #     good_img = Image.open(good_img_path)
            #     good_img = preprocess(good_img)
            #     self.good_imgs.append(good_img)
            #     self.prompts.append(prompt)
            bad_img_paths = glob.glob(train_data_root + path + '/bad/' + '*.png')
            self.bad_imgs_path.extend(bad_img_paths)
            # for bad_img_path in bad_img_paths:
            #     bad_img = Image.open(bad_img_path)
            #     bad_img = preprocess(bad_img)
            #     self.bad_imgs.append(bad_img)
            self.prompts.extend([prompt]*len(good_img_paths))
            self.raw_prompts.extend([raw_prompt]*len(good_img_paths))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        """
        :param item:
        :return:
            img:图片
            txt:图片质量+图片描述
            label:图片质量-good/bad 图片好/坏
            prompt:图片描述-prompt.txt文本内容
        """  
        # return self.good_imgs[item], self.bad_imgs[item], self.prompts[item]
        good_img_path = self.good_imgs_path[item]
        good_img = Image.open(good_img_path)
        good_img = preprocess(good_img)
        
        bad_img_path = self.bad_imgs_path[item]
        bad_img = Image.open(bad_img_path)
        bad_img = preprocess(bad_img)

        return good_img, bad_img, self.prompts[item], self.raw_prompts[item]



# good_dataset = train_data("good", all_good_images_path)  # good图片数据集
# bad_dataset = train_data("bad", all_bad_images_path)  # bad图片数据集
# dataset = good_dataset + bad_dataset  # 形成整个数据集
dataset = train_data(train_data_root)
print(dataset[2])
# print(len(dataset)) # >> 6040对
# train_dataset, val_dataset = dataset[:5000], dataset[5000:]
# train_dataset, val_dataset = train_test_split(dataset, test_size=0, train_size=5000)
train_dataset, val_dataset, _ = random_split(dataset=dataset, lengths=[5000, 1000, 40])#[9664, 2416])
# train_dataset, val_dataset = random_split(dataset=dataset, lengths=[9664, 2416])  # 训练集和验证集划分 4:1
batch_size = 1
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # dataloader batch_size设置的小是因为显存不够
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Loss: InfoNCE
def InfoNCE(pos_logits, neg_logits, anchor_logits, tau: float = 0.8):
    pos_pair = torch.exp(F.cosine_similarity(pos_logits, anchor_logits, dim=-1) / tau)
    neg_pair = torch.exp(F.cosine_similarity(neg_logits, anchor_logits, dim=-1) / tau)
    return -torch.log(pos_pair / (pos_pair + neg_pair))
# loss_img = nn.CrossEntropyLoss().to(device)
# loss_txt = nn.CrossEntropyLoss().to(device)




optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

for i in range(10):
    epoch_loss = 0
    """训练过程"""
    for batch, (good_img, bad_img, prompt, _) in enumerate(train_dataloader):
        print('.', end='')
        prompt_tokens = clip.tokenize(prompt).to(device)
        good_imgs = good_img.to(device)
        logits_per_good_image, logits_per_prompt_for_good = model(good_imgs, prompt_tokens)
        bad_imgs = bad_img.to(device)
        logits_per_bad_image, logits_per_prompt_for_bad = model(bad_imgs, prompt_tokens)

        # if device == "cpu":
        #     ground_truth = torch.arange(1).long().to(device)
        # else:
        #     ground_truth = torch.arange(1, dtype=torch.long, device=device)

        # total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        loss = InfoNCE(logits_per_good_image, logits_per_bad_image, logits_per_prompt_for_good)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    print('epoch %d, loss: %.3f' % (i + 1, epoch_loss/batch_size))

    """验证过程"""
    with torch.no_grad():
        cor = 0
        for batch, (good_img, bad_img, prompt, raw_prompt) in enumerate(train_dataloader):
            # text = clip.tokenize(["a good photo of " + prompt[0], "a bad photo of " + prompt[0]]).to(device)
            # text = clip.tokenize([prompt[0], "other things"]).to(device)
            # text = clip.tokenize(["a photo of " + prompt[0], "a photo without " + prompt[0]]).to(device)
            # text = clip.tokenize(["a good photo of " + prompt[0] + "without other things", "a mess photo of " + prompt[0] + "with other things"]).to(device)
            prompt_tokens = clip.tokenize(prompt).to(device)
            good_imgs = good_img.to(device)
            logits_per_good_image, logits_per_prompt_for_good = model(good_imgs, prompt_tokens)
            bad_imgs = bad_img.to(device)
            logits_per_bad_image, logits_per_prompt_for_bad = model(bad_imgs, prompt_tokens)
            
            if F.cosine_similarity(logits_per_good_image, logits_per_prompt_for_good) >= F.cosine_similarity(logits_per_bad_image, logits_per_prompt_for_bad):
                cor += 1
            else:
                print("%s's prompt is wrong" % (raw_prompt))
            # if device == "cpu":
            #     ground_truth = torch.arange(1).long().to(device)
            # else:
            #     ground_truth = torch.arange(1, dtype=torch.long, device=device)

            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            # pred = np.argmax(probs)
            # """预测[good, bad]的概率[pgood, pbad],如果pgood > pbad 则预测为good, 反之为bad"""
            # if (pred == 0 and label[0] == "good") or (pred == 1 and label[0] == "bad"):
            #     cor += 1
            # else:
            #     print(txt, probs)
        print(cor, '/', val_dataset.__len__(), ' = ', cor / val_dataset.__len__())

    torch.save(model, 'model/model1-%s.pkl' % str(i))
















# for i in range(10):

#     """训练过程"""
#     for batch, (img, txt, label, prompt) in enumerate(train_dataloader):
#         print('.', end='')
#         txts = clip.tokenize(txt).to(device)
#         imgs = img.to(device)
#         logits_per_image, logits_per_text = model(imgs, txts)

#         if device == "cpu":
#             ground_truth = torch.arange(1).long().to(device)
#         else:
#             ground_truth = torch.arange(1, dtype=torch.long, device=device)

#         total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
#         optimizer.zero_grad()
#         total_loss.backward()
#         if device == "cpu":
#             optimizer.step()
#         else:
#             convert_models_to_fp32(model)
#             optimizer.step()
#             clip.model.convert_weights(model)

#     print('[%d] loss: %.3f' % (i + 1, total_loss))

#     """验证过程"""
#     with torch.no_grad():
#         cor = 0
#         for batch, (img, txt, label, prompt) in enumerate(val_dataloader):
#             # text = clip.tokenize(["a good photo of " + prompt[0], "a bad photo of " + prompt[0]]).to(device)
#             # text = clip.tokenize([prompt[0], "other things"]).to(device)
#             # text = clip.tokenize(["a photo of " + prompt[0], "a photo without " + prompt[0]]).to(device)
#             # text = clip.tokenize(["a good photo of " + prompt[0] + "without other things", "a mess photo of " + prompt[0] + "with other things"]).to(device)
#             text = clip.tokenize(["a beautiful and concise photo of " + prompt[0], "a mussy photo of " + prompt[0]]).to(device)
#             image = img.to(device)
#             logits_per_image, logits_per_text = model(image, text)
#             probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#             pred = np.argmax(probs)
#             """预测[good, bad]的概率[pgood, pbad],如果pgood > pbad 则预测为good, 反之为bad"""
#             if (pred == 0 and label[0] == "good") or (pred == 1 and label[0] == "bad"):
#                 cor += 1
#             else:
#                 print(txt, probs)
#         print(cor, '/', val_dataset.__len__(), ' = ', cor / val_dataset.__len__())

#     torch.save(model, 'model/model1-%s.pkl' % str(i))