
import pandas as pd
import torch

def get_X_and_y_tensors(path="sustavianfeed.xlsx", target_column="CRUDE PROTEIN"):
    df=pd.read_excel(path)
    df.set_index("WING TAG", inplace=True)
    df.loc[df["EMPTY MUSCULAR STOMACH"]=="/","EMPTY MUSCULAR STOMACH"]=0
    df["EMPTY MUSCULAR STOMACH"]=df["EMPTY MUSCULAR STOMACH"].astype(float)
    X=df[df.columns.difference([target_column])]
    y=df[target_column]
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    return X,y