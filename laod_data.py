# %%
import pandas as pd
from tqdm.auto import tqdm

columns = ['Ax','Ay','Az','Gx','Gy','Gz', 'Label']

# %%-----------------------------
# 1️⃣ Load UTWENTE SHOAIB dataset
# -------------------------------
progress = tqdm(total=50)
df = pd.DataFrame()
for position in ["Belt", "Left-Pocket", "Right-Pocket", "Upper-Arm", "Wrist"]:
    for i in range(10, 0, -1):
        url = f"UTWENTE SHOAIB/{position}/Participant_{i}.csv.gz"
        p_df = pd.read_csv(url)
        
        df = pd.concat([df, p_df])
        progress.update(1)
df = df.reset_index(drop=True)
progress.close()

# Features and labels
df = df[['Ax','Ay','Az','Gx','Gy','Gz', 'Activity']]
df.columns = columns
df.to_csv('data.csv.gz', index=False, compression='gzip')
df.head()

# %%-----------------------------
# 2️⃣ Load 
# -------------------------------
    