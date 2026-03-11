import sys
import os
import pickle

# 把项目根目录加入 Python 搜索路径
sys.path.append(r"D:\03_Project\Project_python")

instance_name = 'c106_21_beta0.300000_run10'
base_path = r"D:\02_Research\Results\20260307_020517\run_details"
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

print("\n--- scenarios ---")
print(type(data["scenarios"]))
try:
    print("len(scenarios):", len(data["scenarios"]))
except:
    pass