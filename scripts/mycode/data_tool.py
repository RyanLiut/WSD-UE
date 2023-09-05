'''
Some common tools for processing data
'''
from nltk.corpus import wordnet as wn
import sys
sys.path.append("/home/liuzhu/WSD/multilabel-wsd")
from scripts.preprocess.preprocess_sense_embeddings import patched_lemma_from_key

def GT2instkey(GT_dict):
    '''
    Input: GT dict {sentence_id: {instance_ids: {} ...}}
            Ref: all.json (Unified benchmark)
    Output: GT dict {instance_id: [lemma, pos_tag, senses, word]}
    '''

    ins2inf = {vv: [v['lemmas'][int(kk)], v['pos_tags'][int(kk)], v['senses'][kk], v['words'][int(kk)]] for k,v in GT_dict.items() for kk,vv in v['instance_ids'].items()}

    return ins2inf

def inst2lemma(GT_txt):
    '''
    Input: GT text (instance_id GT_lemma_1 GT_lemma_2 ...)
           Ref: ALL.gold.key.txt
    Output: GT dict {instance_id: [lemma_0, lemma_1, ...]}
    '''

    gold = {}
    with open(GT_txt, "r") as f:
        for line in f:
            instance_id, *gold_senses = line.strip().split()
            gold_synsets = [patched_lemma_from_key(s) for s in gold_senses]
            gold[instance_id] = gold_synsets
    
    return gold