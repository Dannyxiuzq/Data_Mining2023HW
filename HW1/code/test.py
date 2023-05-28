import os
import glob
import clip
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F 
import pandas as pd
from model import Net

device = "cuda" if torch.cuda.is_available() else "cpu"
_, preprocess = clip.load("ViT-B/32", device=device)
model = torch.load('D:\DataMine\Data_Mining2023\HW1\code\model\model1-9.pkl')

test_data_root = r'D:\DataMine\Data_Mining2023\HW1\Project_Dataset\Selected_Test_Dataset/'
path_list = os.listdir(test_data_root)
all_image1_path = []
all_image2_path = []
for path in path_list:
    image1_path = glob.glob(test_data_root + path + '/image1/' + '*.png')
    image2_path = glob.glob(test_data_root + path + '/image2/' + '*.png')
    all_image1_path.extend(image1_path)
    all_image2_path.extend(image2_path)

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

class test_data(Dataset):
    def __init__(self, img_path_list):
        self.path = img_path_list

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        raw_img = Image.open(self.path[item])
        img = preprocess(raw_img)
        txt = get_txt_path(os.path.dirname(os.path.dirname(self.path[item]))) + '/prompt.txt'
        file = open(txt, encoding='utf-8')
        prompt = file.read(300)

        # 下面是得到图片的名称
        name = os.path.basename(self.path[item])

        return img, prompt, raw_img, name

image1_dataset = test_data(all_image1_path)
image2_dataset = test_data(all_image2_path)

cor = 0
# with torch.no_grad():
#     """
#     测试过程：
#     取image1和image2的两张对应图片，分别计算[pgood1, pbad1]和[pgood2, pbad2], 如果pgood1 > pgood2 则 image1为good, 反之image2为good
#     """
#     for i in range(image1_dataset.__len__()):
#         img1 = image1_dataset[i][0].unsqueeze(0).to(device)
#         img2 = image2_dataset[i][0].unsqueeze(0).to(device)
#         txt = image1_dataset[i][1]
#         text = clip.tokenize(["a good photo of " + txt, "a bad photo of " + txt]).to(device)
#         image1_logits_per_image, image1_logits_per_text = model(img1, text)
#         image1_probs = image1_logits_per_image.softmax(dim=-1).cpu().numpy()
#         image2_logits_per_image, image2_logits_per_text = model(img2, text)
#         image2_probs = image2_logits_per_image.softmax(dim=-1).cpu().numpy()
#         if image1_probs[0][0] > image2_probs[0][0]:
#             print('y', end='')
#             cor += 1
#         else:
#             print('n', end='')

# print()
# print(cor, image1_dataset.__len__())


# load已经存好的网络
model = torch.load('D:\DataMine\Data_Mining2023\HW1\code\model\model1-1.pkl').to(device)
proj = torch.load('D:\DataMine\Data_Mining2023\HW1\code\model\proj1-1.pkl').to(device)
net = Net(model.encode_image, model.encode_text, proj).to(device)
# 定义一个保存good和bad的列表
image_names = []
image1s = []
image2s = []

with torch.no_grad():
    """
    测试过程：
    取image1和image2的两张对应图片，分别计算[pgood1, pbad1]和[pgood2, pbad2], 如果pgood1 > pgood2 则 image1为good, 反之image2为good
    """
    for i in range(image1_dataset.__len__()):
        img1 = image1_dataset[i][0].unsqueeze(0).to(device)
        img2 = image2_dataset[i][0].unsqueeze(0).to(device)
        txt = image1_dataset[i][1]
        file_name = image1_dataset[i][3]
        text = clip.tokenize("A photo of " + txt + '.').to(device)
        img1_embedding, img2_embedding, prompt_embedding = net(img1, img2, text)
        img1Better = True if F.cosine_similarity(img1_embedding, prompt_embedding) > F.cosine_similarity(img2_embedding, prompt_embedding) else False
        score_img1 = F.cosine_similarity(img1_embedding, prompt_embedding)  # clip img1 score
        score_img2 = F.cosine_similarity(img2_embedding, prompt_embedding)  # clip img2 score
        img1Better = True if score_img1 > score_img2 else False

        image_names.append(file_name)
        image1s.append('good' if img1Better else 'bad')
        image2s.append('bad' if img1Better else 'good')

data = {
    'image_pair_name': image_names,
    'image1': image1s,
    'image2': image2s
}
dataframe = pd.DataFrame(data)
dataframe.to_csv('./test_result.csv', encoding='utf-8')

# print()
# print(cor, image1_dataset.__len__())
