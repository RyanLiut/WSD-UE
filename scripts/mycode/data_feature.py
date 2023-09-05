'''
To extract factors of effecting uncertainties from instances.
Time: 12.13
Author: Z. Liu
'''
import pandas as pd
import json
import numpy as np
from data_tool import GT2instkey
from nltk.corpus import wordnet as wn
from data_tool import inst2lemma
from polyglot.text import Text, Word
from morphemes import Morphemes
from tqdm import tqdm
from transformers import BertTokenizer

data_csvs = [pd.read_csv("results/seminar/all_UE_prediction_MC_T20_seed%d.csv"%i) for i in [10,102,1021]]
GT_dict = json.load(open("data/preprocessed/all/all.json"))
GT_txt = "data/original/all/ALL.gold.key.txt"

GT_dict_byinst = GT2instkey(GT_dict)
GT_dict_inst2lemma = inst2lemma(GT_txt)

df_result = pd.DataFrame()
df_result['instance_id'] = data_csvs[0]['instance_id']
df_result['pred_sense'] = data_csvs[0]['sense_0']
check = {kk:v['senses'][ix][0] for k,v in GT_dict.items() for ix, kk in v['instance_ids'].items() if len(v['instance_ids']) != 0}
df_result['GT_sense'] = [check[i] for i in df_result['instance_id']]

df_result['UE'] = [round(j,4) for j in np.mean([i['SMP_0'].to_numpy() for i in data_csvs], axis=0)]

df_result['pos'] = [GT_dict_byinst[i][1] for i in df_result['instance_id']]
df_result['word'] = [GT_dict_byinst[i][3] for i in df_result['instance_id']]
df_result['lemma'] = [GT_dict_byinst[i][0] for i in df_result['instance_id']]
df_result['nGT'] = [len(GT_dict_byinst[i][2]) for i in df_result['instance_id']]

df_result['word_length'] = [len(i) for i in df_result['word']]

wn_poses = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
df_result['polysemy_degree'] = [len(wn.synsets(i, pos=wn_poses[j])) for i,j in zip(df_result['lemma'], df_result['pos'])]
df_result['synset_size'] = [len(wn.synset(s).lemmas()) for s in df_result['GT_sense']]
df_result['GT_freq'] = [GT_dict_inst2lemma[i][0].count() for i in df_result['instance_id']]

df_result['hyper_depth'] = [len(list(wn.synset(s).closure(lambda x:x.hypernyms()))) for s in df_result['GT_sense']]

df_result['nMorphs'] = [len(Word(w, language="en").morphemes) for w in df_result['word']]
df_result['nMorphs_lemma'] = [len(Word(w, language="en").morphemes) for w in df_result['lemma']]
tok = BertTokenizer.from_pretrained('bert-base-uncased')
df_result['nWP'] = [len(tok.tokenize(w)) for w in df_result['word']]

# m = Morphemes("/home/liuzhu/langauge_resources/MorphoLex-en")
# df_result['nMorphs2'] = [m.parse(w)['morpheme_count'] for w in tqdm(df_result['word'])]
# df_result['nMorphs2_lemma'] = [m.parse(w)['morpheme_count'] for w in tqdm(df_result['lemma'])]
df_result.to_csv("results2/analysis/data.csv", index=False)

df_result_lemmaUE_mean = df_result.groupby(by=['pos', 'lemma', 'GT_sense']).mean().dropna().reset_index().sort_values('UE')
df_result_lemmaUE_std = df_result.groupby(by=['pos', 'lemma', 'GT_sense']).std().dropna().reset_index().sort_values('UE')
df_result_lemmaUE_mean.to_csv("results2/analysis/unique_lemma_mean.csv", index=False)
df_result_lemmaUE_std.to_csv("results2/analysis/unique_lemma_std.csv", index=False)

