from typing import Dict, List, Optional



from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) 读入 summary 文件
path = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Sensitivity\RA\RA.xlsx"
df = pd.read_excel(path)

# 2) 选择要画的指标
value_col = "best_cost_num_charge_opportunistic_mean"   #"best_ra_in_archive_mean"

# 3) 构造网格
grid = df.pivot(index="ra_safe_param", columns="ra_risk_param", values=value_col)
grid = grid.sort_index(axis=0).sort_index(axis=1)

data = grid.values.astype(float)
masked = np.ma.masked_invalid(data)

# 4) colormap，缺失值设为白色
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("white")

# 5) 作图
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest")

# 坐标刻度
x_vals = grid.columns.to_numpy()
y_vals = grid.index.to_numpy()
step = 2   # 每隔2个点显示一次，避免过密

ax.set_xticks(np.arange(0, len(x_vals), step))
ax.set_xticklabels([f"{v:.2f}" for v in x_vals[::step]], rotation=45, ha="right")

ax.set_yticks(np.arange(0, len(y_vals), step))
ax.set_yticklabels([f"{v:.2f}" for v in y_vals[::step]])

# 标签
ax.set_xlabel(r"RA$_{risk}$", fontsize=20, labelpad=12)
ax.set_ylabel(r"RA$_{safe}$", fontsize=20, labelpad=12)
# ax.set_title("Heatmap of best cost in archive", fontsize=20, pad=12)

# 刻度样式
ax.tick_params(axis='both', which='major', labelsize=16, length=6, width=1.2)

# colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of opportunistic-charging events", fontsize=20, labelpad=12)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.show()


