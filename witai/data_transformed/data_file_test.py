import pandas as pd

df = pd.read_json("data.json")
df.head()

if __name__=="__main__": 
    errors = []
    for i,row in df.iterrows():
        print(row["id"], sum([i not in row["text"] for i in row["entities"].values()]))
        if sum([i not in row["text"] for i in row["entities"].values()])>0:
            errors.append([row])

    if errors==[]:
        print("No erros were found")
    if errors!=[]:
        print("Errors are found here:", errors)