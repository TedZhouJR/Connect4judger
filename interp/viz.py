import torch
from opt import parse_args
import matplotlib.pyplot as plt
from data.data import Connect4Dataset
import seaborn as sns
SHOW_INDEX = 3

if __name__ == "__main__":
    args = parse_args()
    heat = torch.load("../result/grad.pt")
    testdataset = Connect4Dataset("../data/test.src.txt", "../data/test.out.txt", False).data
    src = testdataset["Source"]
    tgt = testdataset["Target"]
    i=0
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    for s, t, h in zip(src, tgt, heat):
        if sum(h) < 0.1:
            continue
        h = h / h.sum()
        h = h.reshape((14,6)).transpose(0, 1).flip([0])
        s = s.reshape((14,6)).transpose(0, 1).flip([0])+0.5
        if i == SHOW_INDEX:
            sns.heatmap(h[:,:7].numpy(), ax=ax1, linewidths=.5, vmin=-0.15, vmax=0.25)
            ax1.title.set_text('Attribution of player 1')
            sns.heatmap(h[:,7:].numpy(), ax=ax2, linewidths=.5, vmin=-0.15, vmax=0.25)
            ax2.title.set_text('Attribution of player 2')
            sns.heatmap(s[:,:7].numpy(), ax=ax3, linewidths=.5)
            ax3.title.set_text("Player 1's position")
            sns.heatmap(s[:,7:].numpy(), ax=ax4, linewidths=.5)
            ax4.title.set_text("Player 2's position")
            print(t)
            plt.show()
            break
        i += 1
