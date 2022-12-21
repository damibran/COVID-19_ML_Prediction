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
        self.N = max(img_mat.shape)  # min - edge cut, max - edge extend
        if self.N % 2 != 0:
            self.N += 1

    def getElmnt(self, x, y):
        if (x < self.img_mat.shape[0] and y < self.img_mat.shape[1]):
            return self.img_mat[x][y]
        else:
            return 0

# Реализация из статьи про вычилесние FrEM (https://www.hindawi.com/journals/scn/2020/8822126/)
def frem1(img: image_mat_sampler, t, n, m):

    def Tp(r):
        return (r**(t-1)) * np.sqrt(2/(r**t)) * np.exp(1j*2*n*np.pi*(r**t))

    def x(q):
        return (2*q - img.N + 1)/img.N

    def y(p):
        return (img.N - 1 - 2*p)/img.N

    def r(q, p):
        return np.sqrt(x(q)**2 + y(p)**2)

    def theta(q, p):
        y_v = y(p)
        x_v = x(q)

        angle = np.arctan2(y_v,x_v)

        angle = (2*np.pi + angle) * (angle < 0) + angle*(angle >= 0) # transorm from [-pi;pi] to [0;2pi]

        return angle

    def under_2_sum(q,p):
        return img.getElmnt(q, p) * np.conjugate(Tp(r(q, p))) * np.exp(-1j*m*theta(q, p))

    def under_1_sum(p):
        return sum(under_2_sum(q, p) for q in range(img.N))

    sum_v = sum(under_1_sum(p) for p in range(img.N))

    return sum_v/(np.pi*img.N**2)


if __name__ == '__main__':

    d = xrv.datasets.COVID19_Dataset(
        imgpath="covid-chestxray-dataset/images/", csvpath="covid-chestxray-dataset/metadata.csv")

    Nmax = 9
    alphas = [1.5]
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
    for i in range(163,len(d)):
        print(i)
        sample = d[i]
        sample_img = image_mat_sampler(sample["img"][0])

        for a in alphas:
            for n in range(Nmax):
                res = pool.map(
                    partial(frem1, sample_img, a, n), range(Nmax))
                for m in range(Nmax):
                    val = res[m]
                    data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                         + "_Re"] = np.real(val)
                    data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                         + "_Im"] = np.imag(val)

        data["COVID"] = sample["lab"][3]

        pd.DataFrame(data, index=[i]).to_csv('new_data.csv', mode='a',
                                             index=True, header=not os.path.exists('new_data.csv'))
