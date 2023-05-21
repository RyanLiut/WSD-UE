'''
Data Re-analysis (12.3)
'''
#%%
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn
from evaluate_UE import normalize
from scipy import stats
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from wordcloud import WordCloud, STOPWORDS

seaborn.set()
root = "/home/liuzhu/WSD/multilabel-wsd/"

# %%
# Figure 1
# UE score distribution
df_SR = pd.read_csv(root+"results/seminar/all_UE_prediction_SR_seed1234.csv")
df_MCs = [pd.read_csv(root+"results/seminar/all_UE_prediction_MC_T20_seed%d.csv"%i) for i in [10]]


MPs = df_SR['MP_0'].to_numpy()[[bool(1-i) for i in df_SR['GT_flag'].to_list()]]
SMPs = np.mean([ddf['SMP_0'].to_list() for ddf in df_MCs], axis=0)[[bool(i) for i in df_MCs[0]['GT_flag'].to_list() ]]
PVs = np.mean([ddf['PV_0'].to_list() for ddf in df_MCs], axis=0)[[bool(i) for i in df_MCs[0]['GT_flag'].to_list()]]
BALDs = np.mean([ddf['BALD_0'].to_list() for ddf in df_MCs], axis=0)[[ bool(i) for i in df_MCs[0]['GT_flag'].to_list()]]

fig = plt.figure()
# fig.suptitle("UE score distribution in False-predicted samples.")

ax = plt.subplot(2,2,1)
plt.axvline(np.mean(MPs), color="r", linestyle = "dashed", linewidth=2)
plt.hist(MPs)
plt.text(0.7,600, "s=%s"%str(round(stats.skew(MPs), 2)))
y_vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}'.format(x / 100) for x in y_vals])
plt.title("MP")
plt.xlim([0,1])
plt.xlabel("$ \\times 10^2$")
ax.xaxis.set_label_coords(0.1, 1.1)


ax = plt.subplot(2,2,2)
plt.hist(SMPs)
plt.axvline(np.mean(SMPs), color="r", linestyle = "dashed", linewidth=2)
plt.text(0.7,280, "s=%s"%str(round(stats.skew(SMPs), 2)))
y_vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}'.format(x / 100) for x in y_vals])
plt.title("SMP")
plt.xlim([0,1])
plt.xlabel("$ \\times 10^2$")
ax.xaxis.set_label_coords(0.1, 1.1)

ax = plt.subplot(2,2,3)
plt.hist(normalize(PVs))
plt.axvline(np.mean(normalize(PVs)), color="r", linestyle = "dashed", linewidth=2)
plt.text(0.7,300, "s=%s"%str(round(stats.skew(normalize(PVs)), 2)))
y_vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}'.format(x / 100) for x in y_vals])
plt.title("PV")
plt.xlim([0,1])
plt.xlabel("$ \\times 10^2$")
ax.xaxis.set_label_coords(0.1, 1.1)

ax = plt.subplot(2,2,4)
plt.hist(normalize(BALDs))
plt.axvline(np.mean(normalize(BALDs)), color="r", linestyle = "dashed", linewidth=2)
plt.text(0.7,290, "s=%s"%str(round(stats.skew(normalize(BALDs)), 2)))
y_vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}'.format(x / 100) for x in y_vals])
plt.title("BALD")
plt.xlim([0,1])
plt.xlabel("$ \\times 10^2$")
ax.xaxis.set_label_coords(0.1, 1.1)

# plt.title("UE score distribution in False-predicted samples.")
plt.tight_layout()
# plt.savefig(root+"results2/picInACL/app_com_dist_all.pdf")

# %%
# Figure 2 Update: change the original SemEval-07 to ALL data
# Alearoic uncertainty.
L = [0, 1, 2, 4, 8, 12, 16, 20, 'W']
DP = [1, 2, 3, 4, 5, 6, 7, 8]
L_MC_dfs = [pd.read_csv(root+"results2/DU/PartL%s_ALL_UE_prediction_MC_T20_seed21.csv"%str(i)) for i in L]
L_MP_dfs = [pd.read_csv(root+"results2/DU/PartL%s_ALL_UE_prediction_SR_seed1234.csv"%str(i)) for i in L]

DP_MC_dfs = [pd.read_csv(root+"results2/DU/PartDPN%s_UE_prediction_MC_T20_seed10.csv"%str(i)) for i in DP]
DP_MP_dfs = [pd.read_csv(root+"results2/DU/PartDPN%s_UE_prediction_SR_seed1234.csv"%str(i)) for i in DP]

L_MC_dicts = [json.load(open(root+"results/pesudo_alls/partial_uncertain_2007_S%s.json"%str(i) ))for i in L]
DP_MC_dicts = [json.load(open(root+"results/pesudo_alls/partial_uncertain_2007_DP_N%s_new2.json"%str(i))) for i in DP]


SMP_L_MC = [L_MC_dfs[i]['SMP_0'].mean() for i in range(len(L))]
PV_L_MC = [np.mean(normalize(L_MC_dfs[i]['PV_0'].to_numpy())) for i in range(len(L))]
BALD_L_MC = [np.mean(normalize(L_MC_dfs[i]['BALD_0'].to_numpy())) for i in range(len(L))]
MP_L_SR = [L_MP_dfs[i]['MP_0'].mean() for i in range(len(L))]

SMP_DP_MC = [DP_MC_dfs[i]['SMP_0'].mean() for i in range(len(DP))]
PV_DP_MC = [np.mean(normalize(DP_MC_dfs[i]['PV_0'].to_numpy())) for i in range(len(DP))]
BALD_DP_MC = [np.mean(normalize(DP_MC_dfs[i]['BALD_0'].to_numpy())) for i in range(len(DP))]
MP_DP_SR = [DP_MP_dfs[i]['MP_0'].mean() for i in range(len(DP))]

ACC_L_MC = [1 - L_MC_dfs[i]['Wrong_flag'].mean() for i in range(len(L))]
ACC_DP_MC = [1 - DP_MC_dfs[i]['Wrong_flag'].mean() for i in range(len(DP))]
ACC_L_MP = [1 - L_MP_dfs[i]['Wrong'].mean() for i in range(len(L))]
ACC_DP_MP = [1 - DP_MP_dfs[i]['Wrong'].mean() for i in range(len(DP))]

lentok_L_MC = [np.mean([len(v['words']) for k,v in L_MC_dicts[i].items()]) for i in range(len(L))]
lentok_DP_MC = [np.mean([len(v['words']) for k,v in DP_MC_dicts[i].items()]) for i in range(len(DP))]

#%%
# fig = plt.figure(figsize=(7,5))
# fig.suptitle("SMP")

plt.subplot(2,1,1)
L_inx = [0,1,2,4,5,6,7,8]
DP_inx = [0,1,2,3,4,5,-1]
plt.plot(np.array(L)[L_inx], np.array(SMP_L_MC)[L_inx], "o-", color="tab:blue", label="UE_SMP")
# plt.plot(np.array(L)[L_inx], np.array(PV_L_MC)[L_inx], "s-", color="tab:orange",label="UE_PV")
# plt.plot(np.array(L)[L_inx], np.array(BALD_L_MC)[L_inx], "^-", color="tab:green",label="UE_BALD")
plt.plot(np.array(L)[L_inx], np.array(MP_L_SR)[L_inx], "o-", color="tab:orange", label="UE_MP")
# plt.plot(np.array(L)[L_inx], np.array(ACC_L_MC)[L_inx], "s--", color="tab:blue", label="ACC_SMP")
# plt.plot(np.array(L)[L_inx], np.array(ACC_L_MP)[L_inx], "s--", color="tab:orange", label="ACC_MP")
plt.xlabel("window size: L", loc='right')
# plt.ylabel("value")
plt.legend()
plt.title("(a) Window-controlled")

plt.subplot(2,1,2)
plt.plot(np.array([L[0]]+DP+[L[-1]])[DP_inx], np.array([SMP_L_MC[0]]+SMP_DP_MC+[SMP_L_MC[-1]])[DP_inx], "o-", color="tab:blue", label="UE_SMP")
# plt.plot(np.array([L[0]]+DP+[L[-1]])[DP_inx], np.array([PV_L_MC[0]]+PV_DP_MC+[PV_L_MC[-1]])[DP_inx], "s-", color="tab:orange", label="UE_PV")
# plt.plot(np.array([L[0]]+DP+[L[-1]])[DP_inx], np.array([BALD_L_MC[0]]+BALD_DP_MC+[BALD_L_MC[-1]])[DP_inx], "^-", color="tab:green", label="UE_BALD")
plt.plot(np.array([L[0]]+DP+[L[-1]])[DP_inx], np.array([MP_L_SR[0]]+MP_DP_SR+[MP_L_SR[-1]])[DP_inx], "o-", color="tab:orange", label="UE_MP")
# plt.plot(np.array([L[0]]+DP+[L[-1]])[DP_inx], np.array([ACC_L_MC[0]]+ACC_DP_MC+[ACC_L_MC[-1]])[DP_inx], "s--", color="tab:blue", label="ACC_SMP")
# plt.plot(np.array([L[0]]+DP+[L[-1]])[DP_inx], np.array([ACC_L_MP[0]]+ACC_DP_MP+[ACC_L_MP[-1]])[DP_inx], "s--", color="tab:orange", label="ACC_MP")
plt.xlabel("number of hops: H" )
# plt.ylabel("value")
plt.title("(b) Syntax-controlled")
# plt.legend()

plt.tight_layout()
# plt.savefig(root+"results2/picInACL/app_com_control_re.pdf")

#%%
# Figure 3
# OOD dataset: 42D

df = pd.read_csv(root+"results/seminar/42D_DP_prediction_MCT20.csv")
df_MP = pd.read_csv(root+"results/seminar/42D_DP_prediction_SR.csv")

ACC_OOD = df['GT_flag'].mean()
ACC_OOD_MP = df_MP['GT_flag'].mean()
MP_OOD = [(1-df_MP['score_0'])[[bool(1-j) for j in df_MP['GT_flag'].to_list()]].mean(),(1-df_MP['score_0'])[df_MP['GT_flag']].mean(), (1-df_MP['score_0']).mean()]
SMP_OOD = [(1-df['score_0'])[[bool(1-j) for j in df['GT_flag'].to_list()]].mean(),(1-df['score_0'])[df['GT_flag']].mean(), (1-df['score_0']).mean()] # wrong, right, all
PV_OOD = [normalize(df['var_0'].to_numpy())[[bool(1-j) for j in df['GT_flag'].to_list()]].mean(),normalize(df['var_0'].to_numpy())[df['GT_flag']].mean(), normalize(df['var_0'].to_numpy()).mean()] # wrong, right, all
BALD_OOD = [normalize(df['bald_0'].to_numpy())[[bool(1-j) for j in df['GT_flag'].to_list()]].mean(),normalize(df['bald_0'].to_numpy())[df['GT_flag']].mean(), normalize(df['bald_0'].to_numpy()).mean()] # wrong, right, all
MP_L = [L_MP_dfs[0]['MP_0'][L_MP_dfs[0]['Wrong']].mean(), L_MP_dfs[0]['MP_0'][[bool(1-j) for j in L_MP_dfs[0]['Wrong'].to_list()]].mean()]
SMP_L = [L_MC_dfs[0]['SMP_0'][L_MC_dfs[0]['Wrong_flag']].mean(), L_MC_dfs[0]['SMP_0'][[bool(1-j) for j in L_MC_dfs[0]['Wrong_flag'].to_list()]].mean()]

MP_DP = [DP_MP_dfs[0]['MP_0'][DP_MP_dfs[0]['Wrong']].mean(), DP_MP_dfs[0]['MP_0'][[bool(1-j) for j in DP_MP_dfs[0]['Wrong'].to_list()]].mean()]
SMP_DP = [DP_MC_dfs[0]['SMP_0'][DP_MC_dfs[0]['Wrong_flag']].mean(), DP_MC_dfs[0]['SMP_0'][[bool(1-j) for j in DP_MC_dfs[0]['Wrong_flag'].to_list()]].mean()]
PV_L = [np.mean(normalize(L_MC_dfs[0]['PV_0'][L_MC_dfs[0]['Wrong_flag']].to_numpy())), np.mean(normalize(L_MC_dfs[0]['PV_0'][[bool(1-j) for j in L_MC_dfs[0]['Wrong_flag'].to_list()]].to_numpy()))]
PV_DP = [np.mean(normalize(DP_MC_dfs[0]['PV_0'][DP_MC_dfs[0]['Wrong_flag']].to_numpy())), np.mean(normalize(DP_MC_dfs[0]['PV_0'][[bool(1-j) for j in DP_MC_dfs[0]['Wrong_flag'].to_list()]].to_numpy()))]

BALD_L = [np.mean(normalize(L_MC_dfs[0]['BALD_0'][L_MC_dfs[1]['Wrong_flag']].to_numpy())), np.mean(normalize(L_MC_dfs[0]['BALD_0'][[bool(1-j) for j in L_MC_dfs[0]['Wrong_flag'].to_list()]].to_numpy()))]
BALD_DP = [np.mean(normalize(DP_MC_dfs[0]['BALD_0'][DP_MC_dfs[0]['Wrong_flag']].to_numpy())), np.mean(normalize(DP_MC_dfs[0]['BALD_0'][[bool(1-j) for j in DP_MC_dfs[0]['Wrong_flag'].to_list()]].to_numpy()))]

# acc_list = [ACC_OOD, ACC_L_MC[1], ACC_DP_MP[0]]
# ue_list = [BALD_OOD[-1], BALD_L_MC[1], BALD_DP_MC[0]]
# ue_wrong_list = [BALD_OOD[0], BALD_L[0], BALD_DP[0]]
# ue_right_list = [BALD_OOD[1], BALD_L[1], BALD_DP[1]]
acc_list = [ACC_OOD_MP, ACC_L_MC[0]]#, ACC_DP_MP[0]]
ue_list = [SMP_OOD[-1], SMP_L_MC[0]]#, MP_DP_MC[0]]
ue_wrong_list = [SMP_OOD[0], SMP_L[0]]#, MP_DP[0]]
ue_right_list = [SMP_OOD[1], SMP_L[1]]#, MP_DP[1]]

# plt.plot(acc_list, ue_list, "o--", label="UE")
# plt.plot(acc_list, ue_wrong_list, "s--", label="UE_Wrong")
# plt.plot(acc_list, ue_right_list, "^--", label="UE_Correct")
var = [0.2, 3.7]#, 7.2]
wid = 3.2
plt.bar(np.array(var)-wid/4.5-0.5, ue_right_list, align='center', label='UE_Correct')
plt.bar(np.array(var)-0.5, ue_list, align='center', label='UE')
plt.bar(np.array(var)+wid/4.2-0.5, ue_wrong_list, align='center', label='UE_Wrong')
plt.bar(np.array(var)+wid/2.1-0.47, acc_list, label="ACC")

plt.xticks(var, ['OOD', 'WC w. L=0'])#, 'SC w. H=1'])
# plt.xlabel("CS", loc="right")
plt.ylabel("UE")
# plt.ylim([30, 52])
plt.legend(loc="upper left")
# plt.title("MP score")
# plt.xlabel("ACC")
# plt.ylabel("UE (SMP)")

plt.savefig(root+"results/seminar/summaries/com_twoUncertainties.pdf")

# %%
# Effect 1: pos

pos = ["NOUN", "VERB", "ADJ", "ADV"]
SMP = [0.1272751044351233, 0.21555805616071486, 0.11328026361103284, 0.07733392123878537]
nums = [1881, 948, 507, 184]
pvals = [[1.0, 7.827143378931984e-34, 0.09586348026401191, 0.0001157679189907077], [7.827143378931984e-34, 1.0, 1.7852579276214442e-22, 1.5429234285280304e-18], [0.09586348026401191, 1.7852579276214442e-22, 1.0, 0.007731111206121582], [0.0001157679189907077, 1.5429234285280304e-18, 0.007731111206121582, 1.0]]

ttest = [[0.0, -12.285020044248927, 1.665905623743161, 3.862340957537311], [12.285020044248927, 0.0, 9.916911987071128, 8.940767488465259], [-1.665905623743161, -9.916911987071128, 0.0, 2.671437557832466], [-3.862340957537311, -8.940767488465259, -2.671437557832466, 0.0]]


def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))

    return "n={:d}".format(absolute)

cmap = matplotlib.cm.get_cmap("Blues")
colors = [cmap(0.5), cmap(0.7), cmap(0.3), cmap(0.1)]

ax = plt.subplot(1,2,1)

plt.pie(nums, labels=["UE={:.2f}".format(u) for u in SMP], autopct=lambda pct:func(pct, nums), colors=colors)
plt.legend(labels=pos, loc="upper right", prop={"size":8})

mask=np.triu(np.ones(len(pos), dtype=bool))
# np.fill_diagonal(mask, False)
ax.set_title("(a) UE Distribution")

ax = plt.subplot(1,2,2)

s0 = seaborn.heatmap(ttest, annot=True, xticklabels=pos, yticklabels=pos, fmt=".1f",cbar=False, square=True, center=True, cmap='coolwarm',linewidths=.5)

s0.add_patch(Rectangle((0, 2), 1, 1, fill=False, edgecolor='yellow', lw=2.5))
s0.add_patch(Rectangle((2, 0), 1, 1, fill=False, edgecolor='yellow', lw=2.5))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)
# # fig.suptitle("diff value")   
ax.set_title("(b) Difference Significance", y=1.05)
plt.colorbar(ax.collections[0], cax=cax)
plt.tight_layout()
# plt.show()

plt.savefig(root+"results2/picInACL/effect_POS.pdf", bbox_inches="tight")
# %%
# Effect 2: wordform
df = pd.read_csv(root+"results2/analysis/unique_lemma_mean.csv")
discard_words = ["cos", "sin", "plus", "times"]
df_bylem = df.groupby(by="lemma").mean().reset_index()
pd_bylem = {l:u for l, u in zip(df_bylem['lemma'], df_bylem['polysemy_degree'])}
freq_bylem = {l:u for l, u in zip(df_bylem['lemma'], df_bylem['UE']) if pd_bylem[l]>=3 and l not in discard_words}
print(len(freq_bylem))

# freq_bylem = sorted(freq_bylem.items(), key=lambda x: x[1], reverse=True)
# x_data = [i[0] for i in freq_bylem][:30]
# y_data = [i[1] for i in freq_bylem][:30]
seaborn.set_style("ticks")

ax=plt.subplot(1,2,1)
wordcloud = WordCloud(background_color="white", contour_width=5).generate_from_frequencies(freq_bylem)
plt.imshow(wordcloud)
plt.axis('off')
ax.set_title("(a) Most uncertain lemmas")
# plt.bar(x_data, y_data)

ax=plt.subplot(1,2,2)
wordcloud2 = WordCloud(background_color="white").generate_from_frequencies({k:1-v for k,v in freq_bylem.items()})
plt.imshow(wordcloud2)
plt.axis('off')
ax.set_title("(b) Most certain lemmas")
plt.tight_layout()

plt.savefig(root+"results2/picInACL/effect_lemma.pdf",bbox_inches='tight')
# %%
