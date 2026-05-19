import sys
import os
import pickle

from matplotlib import pyplot as plt

# 把项目根目录加入 Python 搜索路径
sys.path.append(r"D:\03_Project\Project_python")

instance_name = 'c106_21_beta0.300000_run10'
base_path = r"D:\02_Research\6_experimentResults\SPR_parallel\run_details"
file_path = os.path.join(base_path, f"{instance_name}.pkl")

with open(file_path, "rb") as f:
    data = pickle.load(f)

print(type(data))

if isinstance(data, dict):
    for k in data:
        print(k)


print("\n--- archive ---")
print("archive type:", type(data["archive"]))
print("archive len:", len(data["archive"]))

if len(data["archive"]) > 0:
    sol = data["archive"][0]
    print("archive[0] type:", type(sol))
    print("dir(sol):", dir(sol))
    if hasattr(sol, "__dict__"):
        print("sol.__dict__.keys():", sol.__dict__.keys())

print("\n--- key solutions ---")
for k in ["final_best", "min_cost_sol", "min_ra_sol"]:
    print(k, type(data[k]))
    if hasattr(data[k], "__dict__"):
        print(data[k].__dict__.keys())

# print("\n--- scenarios ---")
# print(type(data["scenarios"]))
# try:
#     print("len(scenarios):", len(data["scenarios"]))
# except:
#     pass


files = [
    f"rc106_21_beta0.600000_run{i}.pkl"
    for i in range(1, 11)
]


for fname in files:

    file_path = os.path.join(base_path, fname)

    if not os.path.exists(file_path):
        print("missing:", fname)
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    archive = data["archive"]

    print("\n==============================")
    print("Run:", fname)
    print("Pareto size:", len(archive))
    print("------------------------------")

    for sol in archive:
        print(sol["cost"], sol["ra"])



plt.figure(figsize=(8, 6))

for i, fname in enumerate(files):
    file_path = os.path.join(base_path, fname)

    if not os.path.exists(file_path):
        print(f"File not found: {fname}")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    archive = data["archive"]

    if len(archive) == 0:
        print(f"Empty archive: {fname}")
        continue

    costs = [sol["cost"] for sol in archive]
    ras   = [sol["ra"] for sol in archive]

    points = sorted(zip(costs, ras), key=lambda x: x[0])
    costs = [p[0] for p in points]
    ras   = [p[1] for p in points]

    plt.plot(costs, ras, marker='o', linestyle='-', alpha=0.8, label=f"Run {i+1}")

plt.xlabel("Cost")
plt.ylabel("RA")
plt.title("Pareto Fronts from 10 Runs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()