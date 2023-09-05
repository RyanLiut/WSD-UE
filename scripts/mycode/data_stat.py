'''
Data statistics
'''
#%%
import pandas as pd
from collections import Counter
import json
from nltk.corpus import wordnet as wn
from matplotlib import pyplot as plt
#%%
root = "/home/liuzhu/WSD/multilabel-wsd/"
GT_path = root + "data/original/all/ALL.gold.key.txt"
ALL_path = root + "data/preprocessed/all/all.json"

def tex2dict(text_path):
    # convert a text file into dictionary, with
    # key: instance id
    # value: list of answer key -> lemma (wn)
    gold = {}
    with open(text_path, "r") as f:
        for line in f:
            instance_id, *gold_senses = line.strip().split()
            gold_synsets = [wn.lemma_from_key(s).synset().name() for s in gold_senses]
            gold[instance_id] = gold_synsets
    
    return gold
# Statistics 1: distribution over number of labels
GT_dict = tex2dict(GT_path)
# print(GT_dict)
'''
nlabels_list = [len(v) for k,v in GT_dict.items()]
dist_nlabels = Counter(nlabels_list)
print(dist_nlabels)
'''
#%%
# Restrucuring 1: From dict to human-friendly csv
all_dict = json.load(open(root+'data/preprocessed/all/all.json', 'r'))
df = pd.DataFrame()
df['instance_id'] = [vv for k,v in all_dict.items() for kk,vv in v['instance_ids'].items()]
df['target_word'] = [v['words'][int(kk)] for k,v in all_dict.items() for kk,vv in v['instance_ids'].items()]
df['context'] = [" ".join(v['words']) for k,v in all_dict.items() for kk,vv in v['instance_ids'].items()]
df['GT_sense'] = [GT_dict[i] for i in df['instance_id']]
# df.to_csv("data/preprocessed/all/all.csv", index=False)
#%%
# join pred and GT in some conditions.
# Error cases for miss hit in top 5.
pred = pd.read_csv(root+"results/prediction_top5_SR_MCT20_var_bald.csv")
pred_dict = {i['instance_id']:[i['sense_'+str(j)] for j in range(1)] for _,i in pred.iterrows()}
wrong_flag = [len(set(GT_dict[i]) & set(pred_dict[i])) == 0 for i in pred_dict]
print("The accuracy : {:10.4f}".format(1 - sum(wrong_flag) / len(wrong_flag)))
merged = pd.merge(df, pred, on="instance_id")[wrong_flag]
# merged.to_csv("results/pred_wrong_cases_SR_top1.csv", index=False)

#%%
pred_SR = pd.read_csv(root+"results/prediction_top5_SR.csv")
pred_MC = pd.read_csv(root+"results/prediction_SR_MCT20.csv")
wrong_SR = pd.read_csv(root+"results/pred_wrong_cases_SR_top1.csv")
wrong_MC = pd.read_csv(root+"results/pred_wrong_cases_MCT20_top1.csv")
wrong_SR['score_0'].hist()
plt.title("wrong SR")
wrong_MC['score_0'].hist()
plt.title("wrong MC")
# %%
