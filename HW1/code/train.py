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
learning_rate = 1e-5
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)

"""读取数据集路径"""
train_data_root = r'D:\DataMine\Data_Mining2023\HW1\Project_Dataset\Selected_Train_Dataset/'
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
        if p.grad is not None:
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

class Projection(nn.Module):
    def __init__(self, num_hidden=512) -> None:
        super().__init__()
        self.linear1 = nn.Linear(num_hidden, num_hidden, dtype=torch.float16)
        self.linear2 = nn.Linear(num_hidden, num_hidden, dtype=torch.float16)
        self.activation = F.relu

    def forward(self, embedding):
        return self.linear2(self.activation(self.linear1(embedding)))

# def norm(vec: torch.Tensor):
#     return vec / vec.norm(dim=1, keepdim=True)

class Net(nn.Module):
    def __init__(self, img_encoder, text_encoder, projection) -> None:
        super().__init__()
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.projection = projection

    def forward(self, img_1, img_2, prompt):
        img_1_embedding = self.norm(self.img_encoder(img_1))
        img_2_embedding = self.norm(self.img_encoder(img_2))
        prompt_embedding = self.norm(self.text_encoder(prompt))
        return self.projection(img_1_embedding), self.projection(img_2_embedding), self.projection(prompt_embedding)

    def norm(self, vec: torch.Tensor):
        return vec / vec.norm(dim=1, keepdim=True)

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
        raw_good_img = Image.open(good_img_path)
        good_img = preprocess(raw_good_img)
        
        bad_img_path = self.bad_imgs_path[item]
        raw_bad_img = Image.open(bad_img_path)
        bad_img = preprocess(raw_bad_img)

        return good_img, bad_img, self.prompts[item], self.raw_prompts[item], good_img_path, bad_img_path



# good_dataset = train_data("good", all_good_images_path)  # good图片数据集
# bad_dataset = train_data("bad", all_bad_images_path)  # bad图片数据集
# dataset = good_dataset + bad_dataset  # 形成整个数据集
dataset = train_data(train_data_root)
# print(dataset[2])
# print(len(dataset)) # >> 6040对
# train_dataset, val_dataset = dataset[:5000], dataset[5000:]
# train_dataset, val_dataset = train_test_split(dataset, test_size=0, train_size=5000)
train_dataset, val_dataset, _ = random_split(dataset=dataset, lengths=[5000, 1000, 40])#[9664, 2416])
# train_dataset, val_dataset = random_split(dataset=dataset, lengths=[9664, 2416])  # 训练集和验证集划分 4:1
batch_size = 16
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # dataloader batch_size设置的小是因为显存不够
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

proj = Projection().to(device)  # 用于InfoNCE计算前的embedding的投影函数
net = Net(model.encode_image, model.encode_text, proj).to(device) # 从图片、文本到它们的embedding的网络


# Loss: InfoNCE
def InfoNCE(pos_embeddings, neg_embeddings, anchor_embeddings, tau: float = 0.8):
    # pos_embeddings = proj(pos_embeddings)
    # neg_embeddings = proj(neg_embeddings)
    # anchor_embeddings = proj(anchor_embeddings)
    pos_pair = torch.exp(F.cosine_similarity(pos_embeddings, anchor_embeddings, dim=-1) / tau)
    neg_pair = torch.exp(F.cosine_similarity(neg_embeddings, anchor_embeddings, dim=-1) / tau)
    return -torch.mean(torch.log(pos_pair / (pos_pair + neg_pair)))
# loss_img = nn.CrossEntropyLoss().to(device)
# loss_txt = nn.CrossEntropyLoss().to(device)





optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

for i in range(10):
    epoch_loss = 0
    """训练过程"""
    for batch, (good_img, bad_img, prompt, _, _, _) in enumerate(train_dataloader):
        print('.', end='')
        prompt_tokens = clip.tokenize(prompt).to(device)
        good_imgs = good_img.to(device)
        bad_imgs = bad_img.to(device)
        good_imgs_embedding, bad_imgs_embedding, prompts_embedding = net(good_imgs, bad_imgs, prompt_tokens)
        # total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        loss = InfoNCE(good_imgs_embedding, bad_imgs_embedding, prompts_embedding, proj)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    print('epoch %d, loss: %.3f' % (i + 1, epoch_loss))
    if epoch_loss / batch_size >= 200:
        print("loss is larger than 200")
        if epoch_loss / batch_size >= 10000:
            print("loss is large, flying!")
    elif epoch_loss / batch_size <= 10:
        print("loss is smaller than 10")
        if epoch_loss / batch_size <= 0.5:
            print("loss is small, flying!")
    """验证过程"""
    with torch.no_grad():
        cor = 0
        for batch, (good_img, bad_img, prompt, raw_prompt, _, _) in enumerate(val_dataloader):
            
            prompt_tokens = clip.tokenize(prompt).to(device)
            good_imgs = good_img.to(device)
            bad_imgs = bad_img.to(device)
            good_imgs_embedding, bad_imgs_embedding, prompts_embedding = net(good_imgs, bad_imgs, prompt_tokens)
            
            cor += (F.cosine_similarity(good_imgs_embedding, prompts_embedding) >= F.cosine_similarity(bad_imgs_embedding, prompts_embedding)).sum().item()
            #else:
                #print("%s's prompt is wrong" % str(raw_prompt))
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