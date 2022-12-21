import pandas as pd
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

df = pd.read_csv('abs_data.csv',index_col=0)
scaler = StandardScaler()
df[df.columns[:-1]]=scaler.fit_transform(df[df.columns[:-1]])

D = len(df.columns)-1
N = 100
T = 10000
F = 1  # mutation scaling factor

class Agent:

    def __init__(self) -> None:
        self.x = np.array(Agent.gen_rand_x(),dtype=np.float64)
        self.scores_mean = 0
        self.N_sel=len(df.columns)-1

    def gen_rand_x() -> list[float]:
        x = []
        for col in df.columns[:-1]:
            lb = min(df[col])
            ub = max(df[col])
            x.append(lb+random.random()*(ub-lb))
        return x

    def to_binary(self):
        bin = []
        for x_i in self.x:
            eq = 1/(1+np.exp(-x_i))
            bin.append(eq > 0.5)
        return bin

    def fitness_func(self) -> float:

        bin = self.to_binary()

        N_sel = bin.count(True)
        self.N_sel = N_sel

        to_delete = [df.columns[i] for i in range(len(bin)) if bin[i] == False]

        df1 = df.drop(to_delete, axis='columns')

        X = df1[df1.columns[:-1]]
        Y = df1[df1.columns[-1]]

        #X_train, X_test, Y_train, Y_test = train_test_split(
        #    X,
        #    Y,
        #    test_size=0.2)

        neigh = KNeighborsClassifier(n_neighbors=3)
        
        scores_mean = cross_val_score(neigh,X,Y,).mean()

       # error = (len(y_pred)-np.sum(y_pred == Y_test))/len(y_pred)

        beta = random.random()

        self.scores_mean = scores_mean

        return beta*(1-scores_mean)+(1-beta)*N_sel/D

    def Chain_Cyclone(self, t: float, i: int, x_best: np.array, agents: list['Agent']) -> None:
        r = np.array([random.random() for i in range(len(self.x))])
        if random.random() < 0.5:
            r1 = random.random()
            beta = 2*np.exp(r1*(T-t+1)/T)*np.sin(2*np.pi*r1)
            if (t/T < random.random()):
                # eq 13
                if (i == 0):
                    self.x = x_best+r*(x_best-self.x)+beta*(x_best-self.x)
                else:
                    self.x = x_best+r * \
                        (agents[i-1].x-self.x)+beta*(x_best-self.x)
            else:
                # eq 15
                x_rand = Agent.gen_rand_x()
                if (i == 0):
                    self.x = x_rand+r*(x_rand-self.x)+beta*(x_rand-self.x)
                else:
                    self.x = x_rand+r*(agents[i-1].x-self.x)+beta*(x_rand-self.x)
        else:
            # eq 11
            alpha = 2*np.sqrt(np.log(np.linalg.norm(r)))*r
            if (i == 0):
                self.x = self.x+r*(x_best-self.x)+alpha*(x_best-self.x)
            else:
                self.x = self.x+r*(agents[i-1].x-self.x)+alpha*(x_best-self.x)

    def Somersault(self, x_best: np.array):
        S = 2
        r2 = random.random()
        r3 = random.random()
        self.x = self.x + S*(r2*x_best-r3*self.x)

    def DE(self, i: int, agents: list['Agent']) -> None:
        x_r1 = agents[random.choice(
            [j for j in range(len(agents)) if j != i])].x
        x_r2 = agents[random.choice(
            [j for j in range(len(agents)) if j != i])].x
        x_r3 = agents[random.choice(
            [j for j in range(len(agents)) if j != i])].x

        # probability already satisfied

        V = Agent()
        V.x = x_r1 + F*(x_r2 - x_r3)

        if (V.fitness_func() < self.fitness_func()):
            self.x = V.x


def calc_Pr(i: int, ffs: list[float]) -> float:
    sum = np.sum(ffs)
    return ffs[i]/sum


agents = [Agent() for i in range(N)]
t = 0
best_ind = 0
best_bin = []
while (agents[best_ind].scores_mean < 0.75):
    ffs = [agent.fitness_func() for agent in agents]

    best_ind = np.argmin(ffs)
    best_bin=agents[best_ind].to_binary()
    print('Accuracy',agents[best_ind].scores_mean,'N_sel',agents[best_ind].N_sel)

    for i in range(len(agents)):
        agents[i].Chain_Cyclone(t, i, agents[best_ind].x, agents)

    for i in range(len(agents)):
        Pr = calc_Pr(i, ffs)
        if Pr < 0.5:
            agents[i].Somersault(agents[best_ind].x)
        else:
            agents[i].DE(i, agents)

    t += 1

print(best_bin.count(True))

to_delete = [df.columns[i] for i in range(len(best_bin)) if best_bin[i] == False]

print('Accuracy',agents[best_ind].scores_mean,'N_sel',agents[best_ind].N_sel)

df1 = df.drop(to_delete, axis='columns')

X = df1[df1.columns[:-1]]
Y = df1[df1.columns[-1]]
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)
y_pred = neigh.predict(X_test)

print(classification_report(Y_test, y_pred))
print(accuracy_score(Y_test,y_pred))
