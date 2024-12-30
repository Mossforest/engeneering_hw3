import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tags = ['singlecard', 'ddp', 'paramserver', 'allreduce_ring', 'allreduce_tree', 'selfddp']
path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/engineering_hw3/outputs/official_'
metrics = ['Epoch', 'Batch', 'Batch Time', 'Loss', 'Top1 Accuracy', 'Top5 Accuracy', 'Throughput (samples/s)', 'GPU Utilization (%)', 'Memory Allocated (GB)', 'Memory Reserved (GB)']

# for tag in tags:
tag = 'singlecard'
filepath = path + tag + '/metrics.csv'
df = pd.read_csv(filepath)

selected_metrics = ['Batch Time', 'Loss', 'Throughput (samples/s)'] # , 'Top5 Accuracy', 
selected_name = ['BatchTime', 'Loss', 'Throughput'] # 'Top1Accuracy', 'Top5Accuracy', 

for idx, kk in enumerate(selected_metrics):
    plt.figure()
    # 绘制吞吐量曲线
    x = np.array(df['Batch'][::20])
    y = np.array(df[kk][::20])
    epoch = np.array(df['Epoch'][::20])
    x = x + epoch * 5004

    plt.plot(x, y, linewidth=0.5) # labels=gpu_0, gpu_1
    plt.xlabel('Batch')
    plt.ylabel(kk)
    plt.title(f'singlecard, mean: {np.mean(y):.2f}')
    # plt.ylim(0, 4_000)
    # plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(path + tag + f'/plot_{selected_name[idx]}.png')
    plt.close()



plt.figure()
x = np.array(df['Batch'][::20])
y1 = np.array(df['Top1 Accuracy'][::20])
y2 = np.array(df['Top5 Accuracy'][::20])
epoch = np.array(df['Epoch'][::20])
x = x + epoch * 5004

plt.plot(x, y1, label='Top1 Accuracy', linewidth=0.5) # , gpu_1
plt.plot(x, y2, label='Top5 Accuracy', linewidth=0.5) # , gpu_1
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title(f'singlecard, mean: {np.mean(y1):.2f}, {np.mean(y2):.2f}')
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(path + tag + f'/plot_Accuracy.png')
plt.close()