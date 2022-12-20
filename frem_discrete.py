import scipy
from scipy import integrate
import numpy as np
import torchxrayvision as xrv
import pandas as pd
import random
from multiprocessing import Pool
from functools import partial
import os


def rand() -> float:
    return random.random()*60-30


class image_mat_sampler:
    def __init__(self, img_mat: np.ndarray):
        self.img_mat = img_mat
        self.N = min(img_mat.shape)  # min - edge cut, max - edge extend

    def getElmnt(self, x, y):
        if (x < self.img_mat.shape[0] and y < self.img_mat.shape[1]):
            return self.img_mat[x][y]
        else:
            return 0

# Реализация как в изначальной статье (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0235187#pone.0235187.ref019)


def frem2(img: image_mat_sampler, t, n, m):

    def Enm(r):
        return (r**(t-1))*np.sqrt(2/r**t)*np.exp(-1j*2*n*np.pi*r**t)

    def x(q):
        return (2*q - img.N + 1)/img.N

    def y(p):
        return (img.N - 1 - 2*p)/img.N

    def r(q, p):
        return np.sqrt(x(q)**2 + y(p)**2)

    def theta(q, p):
        return np.arctan(y(p)/x(q))

    def under_2_sum(p, q):
        return img.getElmnt(q, p)*np.conjugate(Enm(r(q, p)))*np.exp(-1j*m*theta(q, p))

    def under_1_sum(p):
        return sum(under_2_sum(p, q) for q in range(0, img.N))

    return (1/(np.pi*img.N**2))*sum(under_1_sum(p) for p in range(0, img.N))

# Реализация из статьи про вычилесние FrEM (https://www.hindawi.com/journals/scn/2020/8822126/)


def frem1(img: image_mat_sampler, t, n, m):

    def An(r):
        return np.sqrt(2/r)*np.exp(1j*2*n*np.pi*r)

    def FrA(r):
        return np.sqrt(t)*(r**(t-1))*An(r**t)

    def x(q):
        return (2*q - img.N + 1)/img.N

    def y(p):
        return (img.N - 1 - 2*p)/img.N

    def r(q, p):
        return np.sqrt(x(q)**2 + y(p)**2)

    def theta(q, p):
        y_v = y(p)
        x_v = x(q)
        if x_v != 0:
            return np.arctan(y_v/x_v)
        elif y_v > 0:
            return np.pi/2
        elif y_v < 0:
            return -np.pi/2
        else:
            return 0

    def under_2_sum(p, q):
        return img.getElmnt(q, p)*np.conjugate(FrA(r(q, p)))*np.exp(-1j*m*theta(q, p))

    def under_1_sum(p):
        return sum(under_2_sum(p, q) for q in range(0, img.N))

    sum_v = sum(under_1_sum(p) for p in range(0, img.N))

    return sum_v/(np.pi*img.N**2)


if __name__ == '__main__':

    d = xrv.datasets.COVID19_Dataset(
        imgpath="covid-chestxray-dataset/images/", csvpath="covid-chestxray-dataset/metadata.csv")

    Nmax = 9
    alphas = [1, 1.2, 1.5]
    data = dict()
    for a in alphas:
        for n in range(Nmax):
            for m in range(Nmax):
                data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                     + "_Re"] = 0
                data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                     + "_Im"] = 0

    data['COVID'] = 0

    pool = Pool()
    for i in range(len(d)):
        sample = d[i]
        sample_img = image_mat_sampler(sample["img"][0])
        for a in alphas:
            for n in range(Nmax):
                res = pool.map(partial(frem1,sample_img,a,n),range(0, Nmax))
                for m in range(Nmax):
                    val = res[m]
                    val = rand()+1j*rand()
                    data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                         + "_Re"] = np.real(val)
                    data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                         + "_Im"] = np.imag(val)
        data["COVID"] = sample["lab"][3]
        pd.DataFrame(data, index=[i]).to_csv('data.csv', mode='a',
                                  index=False, header=not os.path.exists('data.csv'))

    #df = pd.read_csv('data.csv')
