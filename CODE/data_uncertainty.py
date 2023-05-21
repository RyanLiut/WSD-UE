'''
This script generate pesudo sentences to simulate different ranges of uncertainty.
Time: 0222.11.01
'''
#%%
import json
import stanza
from stanza.models.common.doc import Document
from tqdm import tqdm

#%%
root = "/home/liuzhu/WSD/multilabel-wsd/"
all_prev = json.load(open(root+"data/preprocessed/all/all.json", "r"))
all_pesudo_uncertainty = {}

#%%
# context_pesudo, pos_pesudo= ["These", "are", "words"], ["DET", "VERB", "NOUN"]
context_pesudo, pos_pesudo = [], [] # v2

for iid, v in all_prev.items():
    all_pesudo_uncertainty[iid] = {}
    target_inx = [int(kk) for kk in v['instance_ids']]

    all_pesudo_uncertainty[iid]['instance_ids'] = {str(len(context_pesudo) + ii): v['instance_ids'][kk] for ii, kk in enumerate(v['instance_ids'])} 
    all_pesudo_uncertainty[iid]['lemmas'] = context_pesudo + [v['lemmas'][kk] for kk in target_inx]
    all_pesudo_uncertainty[iid]['pos_tags'] = pos_pesudo  + [v['pos_tags'][kk] for kk in target_inx]
    all_pesudo_uncertainty[iid]['senses'] = {str(len(context_pesudo) + ii): v['senses'][kk] for ii, kk in enumerate(v['senses'])}
    all_pesudo_uncertainty[iid]['words'] = context_pesudo + [v['words'][kk] for kk in target_inx]

uncertainty_07 = {k:v for k,v in all_pesudo_uncertainty.items() if k[:11] == "semeval2007"}

# json.dump(uncertainty_07, open("results/pesudo_alls/uncertain_2007_v2.json", "w"))

# partial uncertainty
#%%
# Strategy 1: to set a window of size S
# To be simple, we only take the first instance. (CHANGE to all data)
def findBound(cid, half_wind, max_leng, use_all=False):
    if use_all:
        return (0, cid, max_leng)

    sid = max(cid - half_wind, 0)
    cid = cid - sid
    eid = min(int(sid + 2 * half_wind + 1), max_leng)

    return (sid, cid, eid)
S_list = [0, 1, 2,4,8,12,16,20,'all']
# S_list = [0,1,2]
for S in S_list:
    partial_pesudo_uncertainty = {}

    for iid, v in all_prev.items():
        for insk, insv in v['instance_ids'].items():
            try:
                window_inx = findBound(int(insk), S, len(v['words']), S=="all")
            except:
                print("null instance")
                continue
            partial_pesudo_uncertainty[iid+'.%s'%insk] = {}
            partial_pesudo_uncertainty[iid+'.%s'%insk]['instance_ids'] = {window_inx[1] : insv}
            partial_pesudo_uncertainty[iid+'.%s'%insk]['lemmas'] = v['lemmas'][window_inx[0]:window_inx[2]]
            partial_pesudo_uncertainty[iid+'.%s'%insk]['pos_tags'] = v['pos_tags'][window_inx[0]:window_inx[2]]
            partial_pesudo_uncertainty[iid+'.%s'%insk]['senses'] = {window_inx[1]:insv}
            partial_pesudo_uncertainty[iid+'.%s'%insk]['words'] = v['words'][window_inx[0]:window_inx[2]]

    # partial_uncertainty_07 = {k:v for k,v in partial_pesudo_uncertainty.items() if k[:11] == "semeval2007"}
    # print(len(partial_uncertainty_07))
    if S == "all":
        json.dump(partial_pesudo_uncertainty, open(root+"results/pesudo_alls/lengthALL/partial_uncertain_ALL_SW.json", "w"))
    else:
        json.dump(partial_pesudo_uncertainty, open(root+"results/pesudo_alls/lengthALL/partial_uncertain_ALL_S%i.json"%S, "w"))

#%%
# strategy 2: to extract partial sentences by universial dependency.
n_neighbor_list = [2,3,4,5,6,7,8]

nlp = stanza.Pipeline(lang='en', processors='depparse', depparse_pretagged=True, download_method=False)


def findTargetInx(cid, sent_dict, n_neighbor=1):
    sent = Document([[{'id': i+1, 'text': sent_dict['words'][i], 'lemma': sent_dict['lemmas'][i], 'upos': sent_dict['pos_tags'][i]} for i in range(len(sent_dict['words']))]])
    
    doc = nlp(sent)
    
    candids = [cid]

    for _ in range(n_neighbor):
        hix = [doc.sentences[0].words[int(i)].head - 1 for i in candids]
        tix = [w.id - 1 for w in doc.sentences[0].words if w.head - 1 in candids]
        candids += [i for i in tix + hix]

    candids.append(cid)
    candids = list(set(candids))
    candids = [i for i in candids if i >= 0]
    candids = sorted(candids)

    Centid = candids.index(cid)

    return candids, Centid


for n_neighbor in tqdm(n_neighbor_list):
    partial_pesudo_uncertainty = {}

    for iid, v in all_prev.items():
        for instk, instv in v['instance_ids'].items():
            try:
                targeInx, cid = findTargetInx(int(instk), v, n_neighbor)
            except:
                print("null instance")
                continue
            partial_pesudo_uncertainty[iid+".%s"%instk] = {}
            partial_pesudo_uncertainty[iid+".%s"%instk]['instance_ids'] = {cid : instv}
            try:
                partial_pesudo_uncertainty[iid+".%s"%instk]['lemmas'] = [v['lemmas'][i] for i in targeInx]
                partial_pesudo_uncertainty[iid+".%s"%instk]['pos_tags'] = [v['pos_tags'][i] for i in targeInx]
                partial_pesudo_uncertainty[iid+".%s"%instk]['senses'] = {cid:instv}
                partial_pesudo_uncertainty[iid+".%s"%instk]['words'] = [v['words'][i] for i in targeInx]
            except:
                print("debug")

    # partial_uncertainty_07 = {k:v for k,v in partial_pesudo_uncertainty.items() if k[:11] == "semeval2007"}
    # len_vec = [len(partial_uncertainty_07[i]['words']) for i in partial_uncertainty_07]
    # print(sum(len_vec) / len(len_vec))

    json.dump(partial_pesudo_uncertainty, open(root+"results/pesudo_alls/DPALL/partial_uncertain_ALL_DP_N%i_new2.json"%n_neighbor, "w"))
    # json.dump(partial_uncertainty_07, open(root+"results/pesudo_alls/partial_uncertain_2007_DP_N%i_new2.json"%n_neighbor, "w"))
# %%
