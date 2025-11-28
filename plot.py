import matplotlib.pyplot as plt
import re

# Log data
log_data = """
Epoch 001, Train Loss: 1.7513
Epoch 002, Train Loss: 1.3888, Val Acc: 0.6721
Epoch 003, Train Loss: 1.2249
Epoch 004, Train Loss: 1.1415, Val Acc: 0.6566
Epoch 005, Train Loss: 1.0807
Epoch 006, Train Loss: 1.0417, Val Acc: 0.7785
Epoch 008, Train Loss: 0.9836, Val Acc: 0.7980
Epoch 010, Train Loss: 0.9485, Val Acc: 0.8072
Epoch 012, Train Loss: 0.9125, Val Acc: 0.8128
Epoch 014, Train Loss: 0.8834, Val Acc: 0.8607
Epoch 016, Train Loss: 0.8626, Val Acc: 0.8038
Epoch 018, Train Loss: 0.8529, Val Acc: 0.8488
Epoch 020, Train Loss: 0.8334, Val Acc: 0.8580
Epoch 022, Train Loss: 0.8268, Val Acc: 0.8809
Epoch 024, Train Loss: 0.8081, Val Acc: 0.8717
Epoch 026, Train Loss: 0.8023, Val Acc: 0.8708
Epoch 028, Train Loss: 0.7907, Val Acc: 0.8730
Epoch 030, Train Loss: 0.7743, Val Acc: 0.8814
Epoch 032, Train Loss: 0.7637, Val Acc: 0.8752
Epoch 034, Train Loss: 0.7521, Val Acc: 0.8899
Epoch 036, Train Loss: 0.7403, Val Acc: 0.9058
Epoch 038, Train Loss: 0.7321, Val Acc: 0.9038
Epoch 040, Train Loss: 0.7233, Val Acc: 0.8870
Epoch 042, Train Loss: 0.7140, Val Acc: 0.9201
Epoch 044, Train Loss: 0.7023, Val Acc: 0.9097
Epoch 046, Train Loss: 0.6964, Val Acc: 0.9184
Epoch 048, Train Loss: 0.6851, Val Acc: 0.9089
Epoch 050, Train Loss: 0.6735, Val Acc: 0.9166
Epoch 052, Train Loss: 0.6690, Val Acc: 0.9098
Epoch 054, Train Loss: 0.6606, Val Acc: 0.9256
Epoch 056, Train Loss: 0.6517, Val Acc: 0.9239
Epoch 058, Train Loss: 0.6471, Val Acc: 0.9314
Epoch 060, Train Loss: 0.6328, Val Acc: 0.9308
Epoch 062, Train Loss: 0.6239, Val Acc: 0.9156
Epoch 064, Train Loss: 0.6191, Val Acc: 0.9309
Epoch 066, Train Loss: 0.6129, Val Acc: 0.9369
Epoch 068, Train Loss: 0.6025, Val Acc: 0.9379
Epoch 070, Train Loss: 0.5935, Val Acc: 0.9408
Epoch 072, Train Loss: 0.5888, Val Acc: 0.9415
Epoch 074, Train Loss: 0.5777, Val Acc: 0.9453
Epoch 076, Train Loss: 0.5740, Val Acc: 0.9429
Epoch 078, Train Loss: 0.5697, Val Acc: 0.9487
Epoch 080, Train Loss: 0.5601, Val Acc: 0.9496
Epoch 082, Train Loss: 0.5556, Val Acc: 0.9516
Epoch 084, Train Loss: 0.5522, Val Acc: 0.9513
Epoch 086, Train Loss: 0.5486, Val Acc: 0.9537
Epoch 088, Train Loss: 0.5480, Val Acc: 0.9520
Epoch 090, Train Loss: 0.5424, Val Acc: 0.9554
Epoch 092, Train Loss: 0.5393, Val Acc: 0.9549
Epoch 093, Train Loss: 0.5393, Val Acc: 0.9568
Epoch 095, Train Loss: 0.5392, Val Acc: 0.9569
Epoch 096, Train Loss: 0.5379, Val Acc: 0.9572
Epoch 099, Train Loss: 0.5375, Val Acc: 0.9573
"""

epochs = []
train_losses = []
val_epochs = []
val_accs = []

# Parse Log
for line in log_data.strip().split('\n'):
    ep = int(re.search(r'Epoch (\d+)', line).group(1))
    loss = float(re.search(r'Train Loss: ([\d\.]+)', line).group(1))
    epochs.append(ep)
    train_losses.append(loss)
    if 'Val Acc' in line:
        acc = float(re.search(r'Val Acc: ([\d\.]+)', line).group(1))
        val_epochs.append(ep)
        val_accs.append(acc)

# --- IMPROVEMENT: Global Font Settings ---
# Increase default font sizes to match higher DPI
plt.rcParams.update({'font.size': 16}) 

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Loss (Left Axis)
color = 'tab:red'
ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
ax1.set_ylabel('Train Loss', color=color, fontsize=16, fontweight='bold')
# Increased linewidth to 2.5
ax1.plot(epochs, train_losses, color=color, label='Train Loss', alpha=0.6, linewidth=2.5)
# Increased tick label size
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Plot Accuracy (Right Axis)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation Accuracy', color=color, fontsize=16, fontweight='bold')
# Increased linewidth to 2.5 and markersize to 8
ax2.plot(val_epochs, val_accs, color=color, marker='o', markersize=8, linewidth=2.5, label='Val Acc')
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

# Title
plt.title('Training Loss and Validation Accuracy', fontsize=18, pad=20)

fig.tight_layout()

# --- KEY FIX: Save as PDF or PNG with High DPI ---
# PDF is best for Overleaf (Infinite resolution, clear text)
plt.savefig('./plots/plot.pdf', bbox_inches='tight')

# If you must use PNG, this now looks correct because we scaled up fonts/lines
plt.savefig('./plots/plot.png', dpi=300, bbox_inches='tight')

print("Generated plot.pdf (Best for Overleaf) and plot.png (High Res)")
plt.show()

