import matplotlib.pyplot as plt
import re

# log data
log_data = """
16:48:07 Epoch 001, Train Loss: 1.7331
16:48:20 Epoch 002, Loss: 1.3830, Val Acc: 0.6392, Val F1: 0.6266 (Time: 12.7s) [BEST]
16:48:32 Epoch 003, Train Loss: 1.2188
16:48:45 Epoch 004, Loss: 1.1307, Val Acc: 0.7688, Val F1: 0.7713 (Time: 12.8s) [BEST]
16:48:57 Epoch 005, Train Loss: 1.0764
16:49:10 Epoch 006, Loss: 1.0413, Val Acc: 0.7907, Val F1: 0.7921 (Time: 12.7s) [BEST]
16:49:22 Epoch 007, Train Loss: 1.0096
16:49:36 Epoch 008, Loss: 0.9821, Val Acc: 0.7832, Val F1: 0.7828 (Time: 13.9s)
16:49:49 Epoch 009, Train Loss: 0.9613
16:50:02 Epoch 010, Loss: 0.9447, Val Acc: 0.8268, Val F1: 0.8281 (Time: 13.1s) [BEST]
16:50:14 Epoch 011, Train Loss: 0.9195
16:50:27 Epoch 012, Loss: 0.9127, Val Acc: 0.8161, Val F1: 0.8096 (Time: 13.1s)
16:50:39 Epoch 013, Train Loss: 0.8972
16:50:52 Epoch 014, Loss: 0.8849, Val Acc: 0.8369, Val F1: 0.8370 (Time: 12.7s) [BEST]
16:51:04 Epoch 015, Train Loss: 0.8739
16:51:17 Epoch 016, Loss: 0.8671, Val Acc: 0.8500, Val F1: 0.8490 (Time: 12.7s) [BEST]
16:51:29 Epoch 017, Train Loss: 0.8578
16:51:41 Epoch 018, Loss: 0.8464, Val Acc: 0.8207, Val F1: 0.8207 (Time: 12.7s)
16:51:53 Epoch 019, Train Loss: 0.8384
16:52:06 Epoch 020, Loss: 0.8320, Val Acc: 0.8377, Val F1: 0.8351 (Time: 12.7s)
16:52:18 Epoch 021, Train Loss: 0.8320
16:52:31 Epoch 022, Loss: 0.8208, Val Acc: 0.8804, Val F1: 0.8804 (Time: 12.7s) [BEST]
16:52:43 Epoch 023, Train Loss: 0.8189
16:52:55 Epoch 024, Loss: 0.8114, Val Acc: 0.8628, Val F1: 0.8612 (Time: 12.6s)
16:53:07 Epoch 025, Train Loss: 0.8092
16:53:20 Epoch 026, Loss: 0.7976, Val Acc: 0.8748, Val F1: 0.8758 (Time: 12.6s)
16:53:32 Epoch 027, Train Loss: 0.7989
16:53:45 Epoch 028, Loss: 0.7849, Val Acc: 0.8936, Val F1: 0.8943 (Time: 12.6s) [BEST]
16:53:57 Epoch 029, Train Loss: 0.7858
16:54:10 Epoch 030, Loss: 0.7798, Val Acc: 0.8826, Val F1: 0.8839 (Time: 13.0s)
16:54:22 Epoch 031, Train Loss: 0.7702
16:54:35 Epoch 032, Loss: 0.7653, Val Acc: 0.8922, Val F1: 0.8911 (Time: 13.3s)
16:54:48 Epoch 033, Train Loss: 0.7602
16:55:01 Epoch 034, Loss: 0.7529, Val Acc: 0.8716, Val F1: 0.8712 (Time: 13.2s)
16:55:13 Epoch 035, Train Loss: 0.7423
16:55:26 Epoch 036, Loss: 0.7409, Val Acc: 0.9041, Val F1: 0.9046 (Time: 12.7s) [BEST]
16:55:38 Epoch 037, Train Loss: 0.7338
16:55:51 Epoch 038, Loss: 0.7313, Val Acc: 0.8974, Val F1: 0.8972 (Time: 12.8s)
16:56:03 Epoch 039, Train Loss: 0.7290
16:56:16 Epoch 040, Loss: 0.7234, Val Acc: 0.8991, Val F1: 0.8975 (Time: 12.8s)
16:56:28 Epoch 041, Train Loss: 0.7181
16:56:41 Epoch 042, Loss: 0.7153, Val Acc: 0.9006, Val F1: 0.9004 (Time: 13.5s)
16:56:54 Epoch 043, Train Loss: 0.7085
16:57:08 Epoch 044, Loss: 0.7040, Val Acc: 0.9081, Val F1: 0.9086 (Time: 13.6s) [BEST]
16:57:20 Epoch 045, Train Loss: 0.6948
16:57:33 Epoch 046, Loss: 0.6994, Val Acc: 0.9095, Val F1: 0.9097 (Time: 13.2s) [BEST]
16:57:46 Epoch 047, Train Loss: 0.6897
16:57:59 Epoch 048, Loss: 0.6857, Val Acc: 0.9238, Val F1: 0.9235 (Time: 12.9s) [BEST]
16:58:11 Epoch 049, Train Loss: 0.6836
16:58:23 Epoch 050, Loss: 0.6782, Val Acc: 0.9065, Val F1: 0.9070 (Time: 12.6s)
16:58:35 Epoch 051, Train Loss: 0.6752
16:58:48 Epoch 052, Loss: 0.6705, Val Acc: 0.9234, Val F1: 0.9236 (Time: 12.7s)
16:59:00 Epoch 053, Train Loss: 0.6642
16:59:13 Epoch 054, Loss: 0.6608, Val Acc: 0.9054, Val F1: 0.9060 (Time: 13.0s)
16:59:25 Epoch 055, Train Loss: 0.6577
16:59:38 Epoch 056, Loss: 0.6541, Val Acc: 0.9274, Val F1: 0.9274 (Time: 12.8s) [BEST]
16:59:50 Epoch 057, Train Loss: 0.6463
17:00:03 Epoch 058, Loss: 0.6407, Val Acc: 0.9121, Val F1: 0.9132 (Time: 12.7s)
17:00:15 Epoch 059, Train Loss: 0.6404
17:00:27 Epoch 060, Loss: 0.6336, Val Acc: 0.9285, Val F1: 0.9289 (Time: 12.8s) [BEST]
17:00:40 Epoch 061, Train Loss: 0.6317
17:00:53 Epoch 062, Loss: 0.6275, Val Acc: 0.9322, Val F1: 0.9321 (Time: 13.2s) [BEST]
17:01:05 Epoch 063, Train Loss: 0.6198
17:01:18 Epoch 064, Loss: 0.6195, Val Acc: 0.9276, Val F1: 0.9270 (Time: 13.0s)
17:01:30 Epoch 065, Train Loss: 0.6168
17:01:43 Epoch 066, Loss: 0.6115, Val Acc: 0.9368, Val F1: 0.9370 (Time: 12.7s) [BEST]
17:01:55 Epoch 067, Train Loss: 0.6076
17:02:09 Epoch 068, Loss: 0.6017, Val Acc: 0.9345, Val F1: 0.9345 (Time: 13.1s)
17:02:21 Epoch 069, Train Loss: 0.5999
17:02:33 Epoch 070, Loss: 0.5945, Val Acc: 0.9384, Val F1: 0.9384 (Time: 12.7s) [BEST]
17:02:45 Epoch 071, Train Loss: 0.5923
17:02:58 Epoch 072, Loss: 0.5918, Val Acc: 0.9400, Val F1: 0.9398 (Time: 12.9s) [BEST]
17:03:11 Epoch 073, Train Loss: 0.5841
17:03:24 Epoch 074, Loss: 0.5800, Val Acc: 0.9387, Val F1: 0.9389 (Time: 13.1s)
17:03:36 Epoch 075, Train Loss: 0.5787
17:03:49 Epoch 076, Loss: 0.5752, Val Acc: 0.9484, Val F1: 0.9484 (Time: 13.1s) [BEST]
17:04:02 Epoch 077, Train Loss: 0.5691
17:04:14 Epoch 078, Loss: 0.5659, Val Acc: 0.9470, Val F1: 0.9470 (Time: 12.7s)
17:04:26 Epoch 079, Train Loss: 0.5651
17:04:39 Epoch 080, Loss: 0.5621, Val Acc: 0.9520, Val F1: 0.9521 (Time: 12.7s) [BEST]
17:04:51 Epoch 081, Train Loss: 0.5597
17:05:04 Epoch 082, Loss: 0.5592, Val Acc: 0.9523, Val F1: 0.9523 (Time: 12.7s) [BEST]
17:05:16 Epoch 083, Train Loss: 0.5532
17:05:29 Epoch 084, Loss: 0.5517, Val Acc: 0.9498, Val F1: 0.9497 (Time: 12.6s)
17:05:41 Epoch 085, Train Loss: 0.5515
17:05:54 Epoch 086, Loss: 0.5487, Val Acc: 0.9521, Val F1: 0.9521 (Time: 12.6s)
17:06:06 Epoch 087, Train Loss: 0.5472
17:06:19 Epoch 088, Loss: 0.5466, Val Acc: 0.9548, Val F1: 0.9548 (Time: 12.9s) [BEST]
17:06:31 Epoch 089, Train Loss: 0.5419
17:06:43 Epoch 090, Loss: 0.5424, Val Acc: 0.9558, Val F1: 0.9558 (Time: 12.7s) [BEST]
17:06:56 Epoch 091, Loss: 0.5429, Val Acc: 0.9559, Val F1: 0.9559 (Time: 12.7s) [BEST]
17:07:09 Epoch 092, Loss: 0.5407, Val Acc: 0.9566, Val F1: 0.9565 (Time: 12.7s) [BEST]
17:07:22 Epoch 093, Loss: 0.5397, Val Acc: 0.9568, Val F1: 0.9567 (Time: 13.0s) [BEST]
17:07:35 Epoch 094, Loss: 0.5399, Val Acc: 0.9579, Val F1: 0.9578 (Time: 12.8s) [BEST]
17:07:47 Epoch 095, Loss: 0.5385, Val Acc: 0.9574, Val F1: 0.9574 (Time: 12.7s)
17:08:00 Epoch 096, Loss: 0.5372, Val Acc: 0.9582, Val F1: 0.9582 (Time: 12.9s) [BEST]
17:08:14 Epoch 097, Loss: 0.5364, Val Acc: 0.9588, Val F1: 0.9588 (Time: 13.0s) [BEST]
17:08:26 Epoch 098, Loss: 0.5379, Val Acc: 0.9589, Val F1: 0.9589 (Time: 12.9s) [BEST]
17:08:39 Epoch 099, Loss: 0.5355, Val Acc: 0.9582, Val F1: 0.9581 (Time: 12.8s)
17:08:52 Epoch 100, Loss: 0.5362, Val Acc: 0.9589, Val F1: 0.9588 (Time: 12.6s)
"""

epochs = []
train_losses = []
val_epochs = []
val_accs = []

# 解析 Log
for line in log_data.strip().split('\n'):
    # 提取 Epoch
    ep_match = re.search(r'Epoch (\d+)', line)
    if not ep_match: continue
    ep = int(ep_match.group(1))
    
    # 提取 Loss (兼容 'Train Loss' 和 'Loss')
    loss_match = re.search(r'(?:Train )?Loss: ([\d\.]+)', line)
    if loss_match:
        epochs.append(ep)
        train_losses.append(float(loss_match.group(1)))
    
    # 提取 Val Acc (如果有)
    if 'Val Acc' in line:
        acc_match = re.search(r'Val Acc: ([\d\.]+)', line)
        if acc_match:
            val_epochs.append(ep)
            val_accs.append(float(acc_match.group(1)))

# --- Plot Settings ---
plt.rcParams.update({'font.size': 14}) 

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Loss
color = 'tab:red'
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Train Loss', color=color, fontweight='bold')
ax1.plot(epochs, train_losses, color=color, label='Train Loss', alpha=0.6, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Plot Accuracy
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation Accuracy', color=color, fontweight='bold')
ax2.plot(val_epochs, val_accs, color=color, marker='o', markersize=4, linewidth=2, label='Val Acc')
ax2.tick_params(axis='y', labelcolor=color)

# 标注最高点
if val_accs:
    max_acc = max(val_accs)
    max_ep = val_epochs[val_accs.index(max_acc)]
    ax2.annotate(f'Max: {max_acc:.2%}', xy=(max_ep, max_acc), xytext=(max_ep-20, max_acc-0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Training Loss and Validation Accuracy')
fig.tight_layout()

plt.savefig('./plots/plot.pdf', bbox_inches='tight')
plt.savefig('./plots/plot.png', dpi=300, bbox_inches='tight')
print("Plot generated successfully.")