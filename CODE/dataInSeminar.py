'''
Data shown in the Seminar
'''
#%%
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn

seaborn.set()
root = "/home/liuzhu/WSD/multilabel-wsd/"
#%%
# Table 1

datasets = ['S2', 'S3', 'S7', 'S13', 'S15', "all"]
scores = ['SMP', 'PV', 'BALD']
df = pd.DataFrame()
df_MP = pd.DataFrame()
df_MP['UE Scores'] = ['MP']
df['UE Scores'] = scores

for d in datasets:
    df_list = [pd.read_csv(root+"results/seminar/%s_UE_prediction_MC_T20_seed%d.csv"%(d,i)) for i in [10,102,1021]]

    df[d+"_"+"RCC_AUC_mean"] = [round(np.mean(np.array([i['RCC_AUC_%s'%score][0] for i in df_list]))*100,2) for score in scores]
    df[d+"_"+"RCC_AUC_std"] = [round(np.std(np.array([i['RCC_AUC_%s'%score][0] for i in df_list]))*100,2) for score in scores]
    
    df[d+"_"+"RPP_mean"] = [round(np.mean(np.array([i['RPP_%s'%score][0] for i in df_list]))*100,2) for score in scores]
    df[d+"_"+"RPP_std"] = [round(np.std(np.array([i['RPP_%s'%score][0] for i in df_list]))*100,2) for score in scores]

    df[d+"_"+"CORR"] = [round(np.mean([i['loss'].corr(i[score+"_0"]) for i in df_list]),2) for score in scores]


    df_result = pd.read_csv(root+"results/seminar/%s_UE_prediction_SR_seed1234.csv"%d)
    df_MP[d+"_"+"RCC_AUC_mean"] = [round(df_result['RCC_AUC_MP'][0] * 100, 2)]
    df_MP[d+"_"+"RCC_AUC_std"] = [0]
    df_MP[d+"_"+"RPP_mean"] = [round(df_result['RPP_MP'][0] * 100, 2)]
    df_MP[d+"_"+"RPP_std"] = [0]
    df_MP[d+"_CORR"] = [round(df_result['loss'].corr(df_result['MP_0']),2)]

df_com = pd.concat([df_MP, df], axis=0)
# df_com.to_csv(root+"results/seminar/summaries/UE_scores_2.csv", index=False)


#%%
# Table 2
# For different part-of-speech
all_dict = json.load(open(root+'data/preprocessed/all/all.json'))
pos_list = [v['pos_tags'][int(i)] for k,v in all_dict.items() for i in v['instance_ids']]
pos_inx = {i:[j==i for j in pos_list] for i in ['NOUN', 'VERB', 'ADJ', 'ADV']}

df = pd.DataFrame()
df_MP = pd.DataFrame()
df_MP['UE Scores'] = ['MP']
df['UE Scores'] = scores

d = "all"
pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV']
for pos in pos_list:
    df_list = [pd.read_csv(root+"results/seminar/%s_UE_prediction_MC_T20_seed%d.csv"%(d,i))[pos_inx[pos]].reset_index() for i in [10,102,1021]]

    df[pos+"_"+"RCC_AUC_mean"] = [round(np.mean(np.array([i['RCC_AUC_%s'%score][0] for i in df_list]))*100,2) for score in scores]
    df[pos+"_"+"RCC_AUC_std"] = [round(np.std(np.array([i['RCC_AUC_%s'%score][0] for i in df_list]))*100,2) for score in scores]
    
    df[pos+"_"+"RPP_mean"] = [round(np.mean(np.array([i['RPP_%s'%score][0] for i in df_list]))*100,2) for score in scores]
    df[pos+"_"+"RPP_std"] = [round(np.std(np.array([i['RPP_%s'%score][0] for i in df_list]))*100,2) for score in scores]

    df[pos+"_"+"CORR"] = [round(np.mean([i['loss'].corr(i[score+"_0"]) for i in df_list]),2) for score in scores]


    df_result = pd.read_csv(root+"results/seminar/%s_UE_prediction_SR_seed1234.csv"%d)[pos_inx[pos]].reset_index()
    df_MP[pos+"_"+"RCC_AUC_mean"] = [round(df_result['RCC_AUC_MP'][0] * 100, 2)]
    df_MP[pos+"_"+"RCC_AUC_std"] = [0]
    df_MP[pos+"_"+"RPP_mean"] = [round(df_result['RPP_MP'][0] * 100, 2)]
    df_MP[pos+"_"+"RPP_std"] = [0]
    df_MP[pos+"_CORR"] = [round(df_result['loss'].corr(df_result['MP_0']),2)]

df_com = pd.concat([df_MP, df], axis=0)
# df_com.to_csv(root+"results/seminar/summaries/UE_scores_POS.csv", index=False)

# %%
# Figure 1
# UE score distribution
df_SR = pd.read_csv(root+"results/seminar/all_UE_prediction_SR_seed1234.csv")
df_MCs = [pd.read_csv(root+"results/seminar/all_UE_prediction_MC_T20_seed%d.csv"%i) for i in [10]]


MPs = df_SR['MP_0'].to_numpy()[df_SR['GT_flag'].to_list()]
SMPs = np.mean([ddf['SMP_0'].to_list() for ddf in df_MCs], axis=0)[[bool(1-i) for i in df_MCs[0]['GT_flag'].to_list() ]]
PVs = np.mean([ddf['PV_0'].to_list() for ddf in df_MCs], axis=0)[[bool(1-i) for i in df_MCs[0]['GT_flag'].to_list()]]
BALDs = np.mean([ddf['BALD_0'].to_list() for ddf in df_MCs], axis=0)[[ bool(1-i) for i in df_MCs[0]['GT_flag'].to_list()]]

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

ax = plt.subplot(2,2,1)
plt.hist(MPs)
y_vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}'.format(x / 100) for x in y_vals])
plt.title("MP")
plt.xlim([0,1])
plt.xlabel("$ \\times 10$")
ax.xaxis.set_label_coords(0.1, 1.1)

ax = plt.subplot(2,2,2)
plt.hist(SMPs)
y_vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}'.format(x / 100) for x in y_vals])
plt.title("SMP")
plt.xlim([0,1])
plt.xlabel("$ \\times 10$")
ax.xaxis.set_label_coords(0.1, 1.1)

ax = plt.subplot(2,2,3)
plt.hist(normalize(PVs))
y_vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}'.format(x / 100) for x in y_vals])
plt.title("PV")
plt.xlim([0,1])
plt.xlabel("$ \\times 10$")
ax.xaxis.set_label_coords(0.1, 1.1)

ax = plt.subplot(2,2,4)
plt.hist(normalize(BALDs))
y_vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}'.format(x / 100) for x in y_vals])
plt.title("BALD")
plt.xlim([0,1])
plt.xlabel("$ \\times 10$")
ax.xaxis.set_label_coords(0.1, 1.1)

plt.tight_layout()
plt.savefig(root+"results/seminar/summaries/com_dist.pdf")
# %%
# Figure 2
# Alearoic uncertainty.
L = [0, 1, 2, 4, 8, 12, 16, 20, 'W']
DP = [1, 2, 3, 4, 5, 6, 7, 8]
L_MC_dfs = [pd.read_csv(root+"results/seminar/PartL%s_UE_prediction_MC_T20_seed21.csv"%str(i)) for i in L]
# L_MP_dfs = [pd.read_csv(root+"results/seminar/PartL%d_UE_prediction_SR_seed1234.csv"%i) for i in L]
DP_MC_dfs = [pd.read_csv(root+"results/seminar/PartDPN%s_UE_prediction_MC_T20_seed10.csv"%str(i)) for i in DP]
# DP_MP_dfs = [pd.read_csv(root+"results/seminar/PartDPN%d_UE_prediction_SR_seed1234.csv"%i) for i in DP]
L_MC_dicts = [json.load(open(root+"results/pesudo_alls/partial_uncertain_2007_S%s.json"%str(i) ))for i in L]
DP_MC_dicts = [json.load(open(root+"results/pesudo_alls/partial_uncertain_2007_DP_N%s_new2.json"%str(i))) for i in DP]

for i in DP_MC_dfs:
    if 'GT_flag' in i.columns:
        i['Wrong_flag'] = 1 - i['GT_flag']


SMP_L_MC = [L_MC_dfs[i]['SMP_0'].mean() for i in range(len(L))]
# MP_L_MC = [L_MP_dfs[i]['MP_0'][[bool(1-j) for j in L_MP_dfs[i]['GT_flag'].to_list()]].mean() for i in range(len(L))]
SMP_DP_MC = [DP_MC_dfs[i]['SMP_0'].mean() for i in range(len(DP))]
# MP_DP_MC = [DP_MP_dfs[i]['MP_0'][[bool(1-j) for j in DP_MP_dfs[i]['GT_flag'].to_list()]].mean() for i in range(len(DP))]

ACC_L_MC = [1 - L_MC_dfs[i]['Wrong_flag'].mean() for i in range(len(L))]
ACC_DP_MC = [1 - DP_MC_dfs[i]['Wrong_flag'].mean() for i in range(len(DP))]

lentok_L_MC = [np.mean([len(v['words']) for k,v in L_MC_dicts[i].items()]) for i in range(len(L))]
lentok_DP_MC = [np.mean([len(v['words']) for k,v in DP_MC_dicts[i].items()]) for i in range(len(DP))]

#%%
plt.subplot(2,1,1)
L_inx = [0,1,2,4,5,6,7,8]
DP_inx = [0,1,2,3,4,5,-1]
plt.plot(np.array(L)[L_inx], np.array(SMP_L_MC)[L_inx], "o-", label="UE")
plt.plot(np.array(L)[L_inx], np.array(ACC_L_MC)[L_inx], "s--", label="ACC")
plt.xlabel("window size: L", loc='right')
# plt.ylabel("value")
plt.legend()
plt.title("(a) Window-controlled")

plt.subplot(2,1,2)
plt.plot(np.array([L[0]]+DP+[L[-1]])[DP_inx], np.array([SMP_L_MC[0]]+SMP_DP_MC+[SMP_L_MC[-1]])[DP_inx], "o-", label="UE")
plt.plot(np.array([L[0]]+DP+[L[-1]])[DP_inx], np.array([SMP_L_MC[0]]+ACC_DP_MC+[ACC_L_MC[-1]])[DP_inx], "s--", label="ACC")
plt.xlabel("number of hops: H" ,loc='right')
# plt.ylabel("value")
plt.title("(b) Syntax-controlled")
plt.legend()

plt.tight_layout()
# plt.savefig(root+"results/seminar/summaries/com_control.pdf")

# %%
# Figure 3
# OOD dataset: 42D

df = pd.read_csv(root+"results/42D_DP_prediction_MCT20.csv")
ACC_OOD = df['GT_flag'].mean()
SMP_OOD = [(1-df['score_0'])[[bool(1-j) for j in df['GT_flag'].to_list()]].mean(),(1-df['score_0'])[df['GT_flag']].mean(), (1-df['score_0']).mean()] # wrong, right, all
SMP_L = [L_MC_dfs[1]['SMP_0'][L_MC_dfs[1]['Wrong_flag']].mean(), L_MC_dfs[1]['SMP_0'][[bool(1-j) for j in L_MC_dfs[1]['Wrong_flag'].to_list()]].mean()]
SMP_DP = [DP_MC_dfs[1]['SMP_0'][DP_MC_dfs[1]['Wrong_flag']].mean(), DP_MC_dfs[1]['SMP_0'][[bool(1-j) for j in DP_MC_dfs[1]['Wrong_flag'].to_list()]].mean()]

acc_list = [ACC_OOD, ACC_L_MC[1], ACC_DP_MC[0]]
ue_list = [SMP_OOD[-1], SMP_L_MC[1], SMP_DP_MC[0]]
ue_wrong_list = [SMP_OOD[0], SMP_L[0], SMP_DP[0]]
ue_right_list = [SMP_OOD[1], SMP_L[1], SMP_DP[1]]

# plt.plot(acc_list, ue_list, "o--", label="UE")
# plt.plot(acc_list, ue_wrong_list, "s--", label="UE_Wrong")
# plt.plot(acc_list, ue_right_list, "^--", label="UE_Correct")
var = [0.2, 3.7, 7.2]
wid = 3
plt.bar(np.array(var)-wid/4.5-0.5, ue_right_list, align='center', label='UE_Correct')
plt.bar(np.array(var)-0.5, ue_list, align='center', label='UE')
plt.bar(np.array(var)+wid/4.2-0.5, ue_wrong_list, align='center', label='UE_Wrong')
plt.bar(np.array(var)+wid/2.1-0.5, acc_list, label="ACC")

plt.xticks(var, ['OOD', 'WC w. L=1', 'SC w. H=1'])
# plt.xlabel("CS", loc="right")
plt.ylabel("UE")
# plt.ylim([30, 52])
plt.legend(loc="upper left")
# plt.xlabel("ACC")
# plt.ylabel("UE (SMP)")

plt.savefig(root+"results/seminar/summaries/com_twoUncertainties.pdf")
# %%
