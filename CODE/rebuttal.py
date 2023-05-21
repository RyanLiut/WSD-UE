#%%
import pandas as pd
from sklearn.linear_model import LinearRegression

root = "/home/liuzhu/WSD/multilabel-wsd/"

df = pd.read_csv(root+'results2/analysis/unique_lemma_mean_ACL.csv')
df['pos'] = df['pos'].apply(lambda x: ['NOUN', 'VERB', 'ADJ', 'ADV'].index(x))

print(df.corr())

X = df[df.columns[:-1]].to_numpy()
y = df[df.columns[-1]].to_numpy()

reg = LinearRegression().fit(X, y)
print(reg.coef_)
print(reg.score(X, y))


# %%
