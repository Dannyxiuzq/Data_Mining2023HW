import os
import glob
import clip
import torch
import numpy as np
from CE_score import get_batch_ce_score
from D_score import get_batch_score
import joblib
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim import lr_scheduler
from model import *
"""加载clip模型"""
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-5
#please change if you want to train or not to train
is_train = True
is_eval = True

knn = joblib.load('model\knn.plk')
svm = joblib.load('model\svm.plk')
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)

"""读取数据集路径"""
#please change the path if you want to train
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
train_dataset, val_dataset, _ = random_split(dataset=dataset, lengths=[4800, 1200, 40],generator=torch.manual_seed(42))#[9664, 2416])
# train_dataset, val_dataset = random_split(dataset=dataset, lengths=[9664, 2416])  # 训练集和验证集划分 4:1
batch_size = 16
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # dataloader batch_size设置的小是因为显存不够
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

proj = Projection().to(device)  # 用于InfoNCE计算前的embedding的投影函数
net = Net(model.encode_image, model.encode_text, proj).to(device) # 从图片、文本到它们的embedding的网络
for name, module in net.named_modules():
    print(name, module)

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

if is_eval == True and is_train == False:
    model = torch.load('model\model1-9.pkl').to(device)
    proj = torch.load('model\proj1-1.pkl').to(device)
    net = Net(model.encode_image, model.encode_text, proj).to(device)



optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

# 每个epoch学习率乘以0.5
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

for i in range(10):
    epoch_loss = 0
    """训练过程"""
    if is_train is True:
        for batch, (good_img, bad_img, prompt, _, _, _) in enumerate(train_dataloader):
            print('.', end='')
            prompt_tokens = clip.tokenize(prompt).to(device)
            good_imgs = good_img.to(device)
            bad_imgs = bad_img.to(device)
            good_imgs_embedding, bad_imgs_embedding, prompts_embedding = net(good_imgs, bad_imgs, prompt_tokens)
            # total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            loss = InfoNCE(good_imgs_embedding, bad_imgs_embedding, prompts_embedding, tau=0.4)
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
        #test if the loss is too large or too small
        if epoch_loss  >= 200:
            print("loss is larger than 200")
            if epoch_loss  >= 10000:
                print("loss is large, flying!")
        elif epoch_loss  <= 10:
            print("loss is smaller than 10")
            if epoch_loss  <= 0.5:
                print("loss is small, flying!")
        """验证过程"""
        scheduler.step()
        #print(learning_rate,optimizer.param_groups[0]['lr'])
    if is_eval is True:
        with torch.no_grad():
            cor = 0
            cor_c = 0
            cor_b = 0
            for batch, (good_img, bad_img, prompt, raw_prompt, good_img_path, bad_img_path) in enumerate(val_dataloader):

                prompt_tokens = clip.tokenize(prompt).to(device)
                good_imgs = good_img.to(device)
                bad_imgs = bad_img.to(device)
                good_imgs_embedding, bad_imgs_embedding, prompts_embedding = net(good_imgs, bad_imgs, prompt_tokens)

                good_ce_score = get_batch_ce_score(good_img_path, knn, svm, 3)
                bad_ce_score = get_batch_ce_score(bad_img_path, knn, svm, 3)
                good_ce_score = torch.Tensor(np.array([row[0][1] for row in good_ce_score])).to(device)
                bad_ce_score = torch.Tensor(np.array([row[0][1] for row in bad_ce_score])).to(device)

                cor += (0.95 * F.cosine_similarity(good_imgs_embedding,
                                                   prompts_embedding) + 0.05 * good_ce_score >= 0.95 * F.cosine_similarity(
                    bad_imgs_embedding, prompts_embedding) + 0.05 * bad_ce_score).sum().item()
                cor_c += (F.cosine_similarity(good_imgs_embedding, prompts_embedding) >= F.cosine_similarity(
                    bad_imgs_embedding, prompts_embedding)).sum().item()
                cor_b += (good_ce_score >= bad_ce_score).sum().item()
                print(cor,cor_c,cor_b)
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


                # if you want to see the score of each image, you can uncomment the following code
                # print('--------------------------------------')
                # # print(good_d_score, bad_d_score)
                # print(good_ce_score, bad_ce_score)
                # print('--------------------------------------')
                # print(F.cosine_similarity(good_imgs_embedding, prompts_embedding),
                #       F.cosine_similarity(bad_imgs_embedding, prompts_embedding))
                # print('--------------------------------------')
                # print(0.95 * F.cosine_similarity(good_imgs_embedding, prompts_embedding) + 0.05 * good_ce_score,
                #       0.95 * F.cosine_similarity(bad_imgs_embedding, prompts_embedding) + 0.05 * bad_ce_score)
                # print('--------------------------------------')
            print(cor, '/', val_dataset.__len__(), ' = ', cor / val_dataset.__len__())
            print(cor_c, '/', val_dataset.__len__(), ' = ', cor_c / val_dataset.__len__())
            print(cor_b, '/', val_dataset.__len__(), ' = ', cor_b / val_dataset.__len__())

    # torch.save(model, 'model/net1-%s.pkl' % str(i))# if not want to save model please comment
    # torch.save(proj, 'model/proj1-%s.pkl' % str(i))# if not want to save model please comment
