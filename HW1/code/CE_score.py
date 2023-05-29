import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from skimage.util import view_as_windows
from joblib import Parallel, delayed
'''
def sliding_window(arr, window_size):
    shape = (arr.shape[0] - window_size + 1, arr.shape[1] - window_size + 1, window_size, window_size)
    strides = arr.strides + arr.strides
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    windows = windows.reshape(-1, window_size * window_size)
    return windows
'''


def sliding_window(arr, window_size):
    windows = view_as_windows(arr, (window_size, window_size))
    windows = windows.reshape(-1, window_size * window_size)
    return windows

def permutation_entropy(windows):
    k = windows.shape[0]
    #print(k)
    n = windows.shape[1]
    flat_data = [d.reshape(-1) for d in windows]
    sorted_data = [np.argsort(d) for d in flat_data]
    #print(sorted_data)
    count = {}
    for s in sorted_data:
        si = tuple(s)
        if si in count.keys():
            count[si] += 1
        else:
            count[si] = 1
    #print(math.factorial(n))
    counts = np.zeros((math.factorial(n),), dtype=int)
    t = 0
    for i in count.values():
        counts[t] = i
        t += 1
    p = counts / len(sorted_data)
    #print(p)
    #return (-np.sum(p * np.log2(p + 1e-12))) / np.log2(math.factorial(n))
    return p


def shannon_entropy(p):
    return -np.sum(p * np.log2(p + 1e-12))


def n_shannon_entropy(p):
    n = len(p)
    return (shannon_entropy(p)) / np.log2(n)


def JSdivergence(p):
    n = len(p)
    u = np.full(n, 1 / n)
    h = (p + u) / 2
    JS = shannon_entropy(h) - shannon_entropy(p) / 2 - shannon_entropy(u) / 2
    return JS


def statistical_complexity(p):
    n = len(p)
    d_star = -0.5 * (((n + 1) / n) * np.log2(n) + np.log2(n) - 2 * np.log2(2 * n))
    complexity = JSdivergence(p) * n_shannon_entropy(p) / d_star
    return complexity


def cal_complexity_entropy(img, window_size=3):
    #img = img.convert('L')
    #img_g = np.asarray(img)
    p = permutation_entropy(sliding_window(img, window_size))
    entropy = n_shannon_entropy(p)
    complexity = statistical_complexity(p)
    return complexity, entropy


def cal_complexity_entropy_rgb(img, window_size=3):
    r, g, b = img.split()
    img_r = np.asarray(r)
    img_g = np.asarray(g)
    img_b = np.asarray(b)
    complexity_r, entropy_r = cal_complexity_entropy(img_r, window_size)
    complexity_g, entropy_g = cal_complexity_entropy(img_g, window_size)
    complexity_b, entropy_b = cal_complexity_entropy(img_b, window_size)
    #p = permutation_entropy(sliding_window(img_g, window_size))
    #entropy = n_shannon_entropy(p)
    #complexity = statistical_complexity(p)
    return complexity_r, entropy_r, complexity_g, entropy_g, complexity_b, entropy_b


def get_ce_score(knn, svm, img, window_size=3):
    cr, er, cg, eg, cb, eb = cal_complexity_entropy_rgb(img, window_size)
    c = (cr + cg + cb) / 3
    e = (er + eg + eb) / 3
    #knn_score = knn.predict_proba([[c, e]])
    score = svm.predict_proba([[c, e]])
    #score = (knn_score + svm_score) / 2
    return score


'''
def get_batch_ce_score(batch_path, knn, svm, window_size=3):
    batch_ce_score = []
    for i in range(len(batch_path)):
        img_path = batch_path[i]
        img = Image.open(img_path)
        ce_score = get_ce_score(knn, svm, img, window_size)
        batch_ce_score.append(ce_score)
    return batch_ce_score
'''


def get_batch_ce_score(batch_path, knn, svm, window_size=3):
    # Define a helper function to process a single image
    def process_image(img_path):
        img = Image.open(img_path)
        return get_ce_score(knn, svm, img, window_size)

    # Use parallel processing to process multiple images simultaneously
    batch_ce_score = Parallel(n_jobs=-1)(delayed(process_image)(img_path) for img_path in batch_path)

    return batch_ce_score

def train(dataset, n=35, gamma=0.07, C=20):
    c = []
    e = []
    y = []
    for i in dataset:
        good_img, bad_img, _, prompt = dataset[i]
        goodc_r, goode_r, goodc_g, goode_g, goodc_b, goode_b = cal_complexity_entropy_rgb(good_img, 3)
        badc_r, bade_r, badc_g, bade_g, badc_b, bade_b = cal_complexity_entropy_rgb(bad_img, 3)
        goodc = (goodc_r + goodc_g + goodc_b) / 3
        goode = (goode_r + goode_g + goode_b) / 3
        badc = (badc_r + badc_g + badc_b) / 3
        bade = (bade_r + bade_g + bade_b) / 3
        c.append(goodc)
        c.append(badc)
        e.append(goode)
        e.append(bade)
        y.append(1)
        y.append(0)
    plt.scatter(c, e, c=y)
    plt.show()
    X = [[x1, x2] for x1, x2 in zip(c, e)]
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X, y)
    joblib.dump(knn, 'model/knn.plk')
    svm = SVC(kernel='rbf', gamma=gamma, C=C, probability=True)
    svm.fit(X, y)
    joblib.dump(svm, 'model/svm.plk')


#img_path = "Project_Dataset/Selected_Train_Dataset/a yellow t-shirt with a dog on it_/bad/a-yellow-t-shirt-with-a-dog-on-3.png"
#img = Image.open(img_path)
#img1 = img.convert('L')
#gray_array = np.asarray(img1)
#print(n_shannon_entropy(permutation_entropy(sliding_window(gray_array, 2))))
#t = np.asarray([[6, 0, 2], [4, 5, 2], [6, 7, 4]])
#print(permutation_entropy(sliding_window(t, 2)))
#print(cal_complexity_entropy(img))
#print(cal_complexity_entropy_rgb(img))
