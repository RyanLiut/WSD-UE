'''
ACC/Risk performance for choosing samples by UE.
Time: 22/10/31
'''
import numpy as np
from nltk.corpus import wordnet as wn

# Ref: https://github.com/AIRI-Institute/uncertainty_transformers/blob/510b5770d80302f5ffabd9b059f9000c8cfe370c/src/analyze_results.py#L240
def get_rcc_auc(ue_score, risk, return_points=False):
    # risk-coverage curve's area under curve
    conf = [-i for i in ue_score]
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=True)

    cumulative_risk = [cr_pair[0][1]]
    for i in range(1, n):
        cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])

    points_x = []
    points_y = []

    auc = 0
    for k in range(n):
        auc += cumulative_risk[k] / (1 + k)
        points_x.append((1 + k) / n)  # coverage
        points_y.append(cumulative_risk[k] / (1 + k))  # current avg. risk

    auc /= n
    if return_points:
        return auc, points_x, points_y
    else:
        return auc

def tex2dict(text_path):
    gold = {}
    with open(text_path, "r") as f:
        for line in f:
            instance_id, *gold_senses = line.strip().split()
            gold_synsets = [wn.lemma_from_key(s).synset().name() for s in gold_senses]
            gold[instance_id] = gold_synsets
    
    return gold

def get_risk_binaries(pred_list, GT_path): 
    # pred_list: [(instance_id, pred_synset), .., ()]
    GT_dict = tex2dict(GT_path)
    risk_binaries = [len(set(GT_dict[i[0]]) & set([i[1]])) == 0 for i in pred_list]

    return risk_binaries

def get_rpp(ue_score, loss):
    ue_matrix = np.array(ue_score).reshape(-1,1) - np.array(ue_score).reshape(1,-1)
    loss_matrix = np.array(loss).reshape(-1,1) - np.array(loss).reshape(1,-1)
    rpp = np.sum(ue_matrix * loss_matrix < 0) / len(loss) ** 2

    return rpp

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))
    