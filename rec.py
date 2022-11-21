bleus = [0.06553491492713884, 0.08700886028937257, 0.09774709100952941, 0.12630731178514235, 0.15885119008903567,
         0.18558153660586252, 0.19878771217143246, 0.20509285997322452, 0.22258337471035722, 0.2353288785418157,
         0.24411578013749707, 0.25605673194986284, 0.25920480837772697, 0.270788568952155, 0.27492220589243216,
         0.283736606295495, 0.2436004750534288, 0.2794589132309532, 0.280857916739201, 0.28423186339194884]

rouges = [{'rouge-1': 0.2927121476740946, 'rouge-2': 0.10845202814117953, 'rouge-l': 0.28058951699180096},
          {'rouge-1': 0.3857199359379728, 'rouge-2': 0.131857712760532, 'rouge-l': 0.3697281799146959},
          {'rouge-1': 0.4210779542838079, 'rouge-2': 0.14855015478491615, 'rouge-l': 0.40257187463284183},
          {'rouge-1': 0.46909833047698746, 'rouge-2': 0.18868178817220435, 'rouge-l': 0.45074880392921374},
          {'rouge-1': 0.5075415968516556, 'rouge-2': 0.22592048682142782, 'rouge-l': 0.4890807350544113},
          {'rouge-1': 0.5335331175229497, 'rouge-2': 0.2570357656743692, 'rouge-l': 0.5157323179687927},
          {'rouge-1': 0.5479552981234407, 'rouge-2': 0.27132565369545486, 'rouge-l': 0.5293726345716834},
          {'rouge-1': 0.5505903089836988, 'rouge-2': 0.2801836045230783, 'rouge-l': 0.53302038088754},
          {'rouge-1': 0.5706945961132291, 'rouge-2': 0.3006498912578591, 'rouge-l': 0.5539331423197492},
          {'rouge-1': 0.5813996411700605, 'rouge-2': 0.31385054538816265, 'rouge-l': 0.5646227452542831},
          {'rouge-1': 0.5880890934415329, 'rouge-2': 0.3222242121419705, 'rouge-l': 0.5715093108082531},
          {'rouge-1': 0.5988720197555084, 'rouge-2': 0.3374608617212454, 'rouge-l': 0.5822949751721727},
          {'rouge-1': 0.6042873453754639, 'rouge-2': 0.34053149185853515, 'rouge-l': 0.5880439395514558},
          {'rouge-1': 0.6124786617644912, 'rouge-2': 0.35372838915241095, 'rouge-l': 0.5966862261479158},
          {'rouge-1': 0.616651354042288, 'rouge-2': 0.35995938469444505, 'rouge-l': 0.6007432985281458},
          {'rouge-1': 0.6206396418248799, 'rouge-2': 0.36719382805067496, 'rouge-l': 0.6058642340790074},
          {'rouge-1': 0.5936959296819767, 'rouge-2': 0.32633782117729143, 'rouge-l': 0.5765991879508504},
          {'rouge-1': 0.6219320533868236, 'rouge-2': 0.3633191533636166, 'rouge-l': 0.6067004024063075},
          {'rouge-1': 0.6229089917061298, 'rouge-2': 0.36676556248531234, 'rouge-l': 0.606996213264591},
          {'rouge-1': 0.6236758713588123, 'rouge-2': 0.3693627291783766, 'rouge-l': 0.607810131075441}]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv
import seaborn as sns

sns.set_style("ticks")


def tensorboard_smoothing(x, smooth=0.6):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):
        x[i] = (x[i - 1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x


def assb(arr, x_label='Epochs', y_label='Score'):
    x = []
    y = []
    i = 0
    for row in arr:
        x.append(i)
        i += 1
        y.append(float(row))  # 强制转换

    smooth_y = tensorboard_smoothing(y)
    d = {x_label: x, y_label: y, 'smooth': smooth_y}
    return pd.DataFrame(d)


if __name__ == "__main__":
    b, r1, r2, rl = [], [], [], []

    for i in bleus:
        b.append(i)

    for i in rouges:
        r1.append(i['rouge-1'])
        r2.append(i['rouge-2'])
        rl.append(i['rouge-l'])


    b = assb(b)
    r1 = assb(r1)
    r2 = assb(r2)
    rl = assb(rl)

    plt.figure(figsize=(7, 6))
    plt.subplot(2, 1, 1)
    sns.lineplot(x="Epochs", y="Score", data=b, label="BLEU Score")
    # sns.lineplot(x="Epochs", y="smooth", data=b4_xe)
    # sns.lineplot(x="Epochs", y="smooth", data=b4_rl)
    plt.fill_between(b['Epochs'], b['Score'], b['smooth'], facecolor='blue', alpha=0.2)
    plt.title("BLEU Metric")
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.subplot(2, 1, 2)
    sns.lineplot(x="Epochs", y="Score", data=r1, label="ROUGE-1 Score")
    sns.lineplot(x="Epochs", y="Score", data=r2, label="ROUGE-2 Score")
    sns.lineplot(x="Epochs", y="Score", data=rl, label="ROUGE-L Score")
    plt.fill_between(r1['Epochs'], r1['Score'], r1['smooth'], facecolor='blue', alpha=0.2)
    plt.fill_between(r2['Epochs'], r2['Score'], r2['smooth'], facecolor='orange', alpha=0.5)
    plt.fill_between(rl['Epochs'], rl['Score'], rl['smooth'], facecolor='pink', alpha=0.5)
    plt.title("ROUGE Metric")
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)


    plt.tight_layout()

    plt.savefig(fname="metrics.svg", format="svg")
    plt.show()
