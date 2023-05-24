import os
import glob
import clip
import torch
from torch.utils.data import Dataset
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
_, preprocess = clip.load("ViT-B/32", device=device)
model = torch.load('D:\DataMine\Data_Mining2023\HW1\code\model\model1-4.pkl')

test_data_root = r'D:\DataMine\Data_Mining2023\HW1\Project_Dataset\Selected_Test_Dataset'
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
        txt = get_txt_path(self.path[item]) + 'prompt.txt'
        file = open(txt, encoding='utf-8')
        prompt = file.read(300)
        return img, prompt, raw_img

image1_dataset = test_data(all_image1_path)
image2_dataset = test_data(all_image2_path)

cor = 0
with torch.no_grad():
    """
    测试过程：
    取image1和image2的两张对应图片，分别计算[pgood1, pbad1]和[pgood2, pbad2], 如果pgood1 > pgood2 则 image1为good, 反之image2为good
    """
    for i in range(image1_dataset.__len__()):
        img1 = image1_dataset[i][0].unsqueeze(0).to(device)
        img2 = image2_dataset[i][0].unsqueeze(0).to(device)
        txt = image1_dataset[i][1]
        text = clip.tokenize(["a good photo of " + txt, "a bad photo of " + txt]).to(device)
        image1_logits_per_image, image1_logits_per_text = model(img1, text)
        image1_probs = image1_logits_per_image.softmax(dim=-1).cpu().numpy()
        image2_logits_per_image, image2_logits_per_text = model(img2, text)
        image2_probs = image2_logits_per_image.softmax(dim=-1).cpu().numpy()
        if image1_probs[0][0] > image2_probs[0][0]:
            print('y', end='')
            cor += 1
        else:
            print('n', end='')

print()
print(cor, image1_dataset.__len__())
