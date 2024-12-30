import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tags = ['ddp', 'paramserver', 'allreduce_ring', 'allreduce_tree', 'selfddp']
path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/engineering_hw3/outputs/official_'
metrics = ['Epoch', 'Batch', 'Batch Time', 'Loss', 'Top1 Accuracy', 'Top5 Accuracy', 'Throughput (samples/s)', 'GPU Utilization (%)', 'Memory Allocated (GB)', 'Memory Reserved (GB)']

# for tag in tags:
tag = 'ddp'  # 'paramserver'
filepath_list = [path + tag + f'/metrics_{gpu}.csv' for gpu in range(4)]
df_list = [pd.read_csv(filepath) for filepath in filepath_list]

selected_metrics = ['Batch Time', 'Loss', 'Throughput (samples/s)'] # , 'Top5 Accuracy', 
selected_name = ['BatchTime', 'Loss', 'Throughput'] # 'Top1Accuracy', 'Top5Accuracy', 

for idx, kk in enumerate(selected_metrics):
    plt.figure()
    # 绘制吞吐量曲线
    x = np.array(df_list[0]['Batch'][::20])
    y_list = [np.array(df[kk][::20]) for df in df_list]
    epoch = np.array(df_list[0]['Epoch'][::20])
    x = x + epoch * 5004
    if kk == 'Throughput (samples/s)':
        sum_y = np.sum(y_list, axis=0)
    else:
        sum_y = np.mean(y_list, axis=0)

    for gpu, y in enumerate(y_list):
        plt.plot(x, y, label=f'gpu_{gpu}', linewidth=0.5) # labels=gpu_0, gpu_1
    plt.plot(x, sum_y, label=f'sum', linewidth=0.5) # labels=gpu_0, gpu_1
    plt.xlabel('Batch')
    plt.ylabel(kk)
    plt.title(f'{tag}, mean: {np.mean(sum_y):.2f}')
    # plt.ylim(0, 4_000)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(path + tag + f'/plot_{selected_name[idx]}.png')
    plt.close()



plt.figure()
x = np.array(df_list[0]['Batch'][::20])
y1_list = [np.array(df['Top1 Accuracy'][::20]) for df in df_list]
y2_list = [np.array(df['Top5 Accuracy'][::20]) for df in df_list]
epoch = np.array(df_list[0]['Epoch'][::20])
x = x + epoch * 5004
avg_y1 = np.mean(y1_list, axis=0)
avg_y2 = np.mean(y2_list, axis=0)

plt.plot(x, avg_y1, label='Top1 Accuracy', linewidth=0.5) # , gpu_1
plt.plot(x, avg_y2, label='Top5 Accuracy', linewidth=0.5) # , gpu_1
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title(f'{tag}, mean: {np.mean(avg_y1):.2f}, {np.mean(avg_y2):.2f}')
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(path + tag + f'/plot_Accuracy.png')
plt.close()