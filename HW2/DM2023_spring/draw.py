import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
df = pd.read_csv('./dataset/BekaaValley_train.csv')
min_num = 0
max_num = 30
num = 30
interval = (max_num-min_num)/num
lower_bounds = np.linspace(min_num, max_num, num=num)
prob_daji = []
prob_zhencha = []
prob_zhihui = []
prob_ganrao = []
prob_yourao = []
for lower_bound in lower_bounds:
    up_bound = lower_bound + interval
    prob_daji.append(len((df.loc[(df['purpose'] == '打击') & (df['group'] > lower_bound) & (df['group'] < up_bound)])) / len(df.loc[(df['purpose'] == '打击')]))
    prob_zhencha.append(len((df.loc[(df['purpose'] == '侦察') & (df['group'] > lower_bound) & (df['group'] < up_bound)])) / len(df.loc[(df['purpose'] == '侦察')]))
    prob_zhihui.append(len((df.loc[(df['purpose'] == '指挥') & (df['group'] > lower_bound) & (df['group'] < up_bound)])) / len(df.loc[(df['purpose'] == '指挥')]))
    prob_ganrao.append(len((df.loc[(df['purpose'] == '干扰') & (df['group'] > lower_bound) & (df['group'] < up_bound)])) / len(df.loc[(df['purpose'] == '干扰')]))
    prob_yourao.append(len((df.loc[(df['purpose'] == '诱扰') & (df['group'] > lower_bound) & (df['group'] < up_bound)])) / len(df.loc[(df['purpose'] == '诱扰')]))
plt.plot(lower_bounds, prob_daji, label='打击')
plt.plot(lower_bounds, prob_zhencha, label='侦察')
plt.plot(lower_bounds, prob_zhihui, label='指挥')
plt.plot(lower_bounds, prob_ganrao, label='干扰')
plt.plot(lower_bounds, prob_yourao, label='诱扰')
plt.title('group')
plt.xlabel('group value')
plt.ylabel('probability')
plt.legend()
plt.savefig(r'D:\DataMine\Data_Mining2023\HW2\DM2023_spring\group.png')
plt.show()