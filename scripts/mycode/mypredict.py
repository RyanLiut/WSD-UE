#%%
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch_scatter import scatter_mean

from nltk.corpus import wordnet as wn
from tqdm import tqdm

import sys
sys.path.append('/home/liuzhu/WSD/multilabel-wsd')
root = "/home/liuzhu/WSD/multilabel-wsd/"


#%%
from wsd.data.dataset import WordSenseDisambiguationDataset
from wsd.data.processor import Processor
from wsd.models.model import SimpleModel
import pandas as pd
from evaluate_UE import get_rcc_auc, get_risk_binaries, get_rpp, normalize
from bertviz import model_view, head_view

import numpy as np
import random

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--processor', type=str, default=root+'bert-large/processor_config.json')
    parser.add_argument('--model', type=str, default=root+'bert-large/best_checkpoint_val_f1=0.7626_epoch=018.ckpt')
    parser.add_argument('--model_input', type=str, default=root+'data/preprocessed/semeval2007/semeval2007.json')
    parser.add_argument('--model_output_dir', type=str, default=root+'results4')
    parser.add_argument('--evaluation_input', type=str, default=root+'data/original/semeval2007/semeval2007.gold.key.txt')

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_Tsamples', type=int, default=None)
    parser.add_argument('--rand_seed', default=1234, type=int)
    parser.add_argument('--mark', type=str, default=None)

    # Store the arguments in hparams.
    # args = parser.parse_args()

    args, unknown = parser.parse_known_args()
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    processor = Processor.from_config(args.processor)

    test_dataset = WordSenseDisambiguationDataset(args.model_input)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences,
        shuffle = False) # To evaulate

    model = SimpleModel.load_from_checkpoint(args.model)
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model.to(device)
    if args.n_Tsamples:
        model.train()
        n_Tsamples = args.n_Tsamples
    else:
        model.eval()
        n_Tsamples = 1

    predictions_scores_T = []

    print("********MARK: %s**********" % args.mark)
    print("MODEL INPUT: %s"%args.model_input)

    print("---------------Model Evaluation, %d samples---------------" % n_Tsamples)
    for t in tqdm(range(n_Tsamples)):
        predictions = {}
        predictions_topK = {}
        predictions_scores = {}
        candidate_synsets = {}
        golds_inx = {}

        weighted_topK = {}

        with torch.no_grad():
            ii = 1
            for x, _ in tqdm(test_dataloader):
                x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
                y = model(x)
                wid2str = {test_dataset.sentences[i]['sentence_id']: test_dataset.sentences[i]['words'] for i in range(len(test_dataset.sentences))}
                batch_predictions, batch_predictions_topK, batch_predictions_scores, batch_candidate_synsets, batch_golds_inx = processor.decode(x, y, args.evaluation_input)
                predictions.update(batch_predictions)
                predictions_topK.update(batch_predictions_topK)
                predictions_scores.update(batch_predictions_scores)
                candidate_synsets.update(batch_candidate_synsets)
                golds_inx.update(batch_golds_inx)
        
        predictions = sorted(list(predictions.items()), key=lambda kv: kv[0])
        predictions_topK = sorted(list(predictions_topK.items()), key=lambda kv: kv[0])
        predictions_scores_T.append(predictions_scores)

    prediction_matrices = {k:torch.tensor([sample[k].tolist() for sample in predictions_scores_T]) for k in predictions_scores}
    predictions_scores_mean = {k:v.mean(dim=0) for k,v in prediction_matrices.items()}
    predictions_scores_var = {k:v.var(dim=0) for k,v in prediction_matrices.items()}
    predictions_scores_bald = {k: -(predictions_scores_mean[k]*predictions_scores_mean[k].log()).sum() + (v*v.log()).sum()/v.shape[0] for k,v in prediction_matrices.items()}
    
    predictions_scores = sorted(list(predictions_scores.items()), key=lambda kv: kv[0])
    predictions_scores_mean = sorted(list(predictions_scores_mean.items()), key=lambda kv: kv[0])
    predictions_scores_var= sorted(list(predictions_scores_var.items()), key=lambda kv: kv[0])

    # To evaluate UE scores.
    print("---------------Uncerainty Estimation---------------")
    risk_binaries = {}
    # GT_path = "data/original/all/ALL.gold.key.txt"
    GT_path = args.evaluation_input
    best_inx = [k[1].argmax() for k in predictions_scores_mean]
    MC_predictions = {k[0]:candidate_synsets[k[0]][best_inx[i]] for i,k in enumerate(predictions_scores_mean)}
    MC_predictions = sorted(list(MC_predictions.items()), key=lambda kv: kv[0])
    risk_binaries['SR'] = get_risk_binaries(predictions, GT_path) # items should be corresponding!
    risk_binaries['MC'] = get_risk_binaries(MC_predictions, GT_path)
    print("The acc of SR: {:10.4f}".format(1- sum(risk_binaries['SR'])/len(risk_binaries['SR'])))
    print("The acc of MC: {:10.4f}".format(1- sum(risk_binaries['MC'])/len(risk_binaries['MC'])))

    rcc_auc = {}
    rcc_auc['MP'] = round(get_rcc_auc([1 - k[1].max().item() for k in predictions_scores], risk_binaries['SR']), 4)
    if n_Tsamples != 1:
        rcc_auc['SMP'] = round(get_rcc_auc([1 - k[1][best_inx[i]].item() for i,k in enumerate(predictions_scores_mean)], risk_binaries['MC']),4)
        rcc_auc['PV'] = round(get_rcc_auc([k[1][best_inx[i]].item() for i,k in enumerate(predictions_scores_var)], risk_binaries['MC']),4)
        rcc_auc['BALD'] = round(get_rcc_auc([predictions_scores_bald[k[0]] for k in predictions_scores_mean], risk_binaries['MC']),4)
    print("RCC_AUC: %s" % str(rcc_auc))

    rpp = {}
    rpp['MP'] = round(get_rpp([1 - k[1].max().item() for k in predictions_scores], [-k[1][golds_inx[k[0]]].item() for k in predictions_scores]),4)
    if n_Tsamples != 1:
        rpp['SMP'] = round(get_rpp([1 - k[1][best_inx[i]].item() for i,k in enumerate(predictions_scores_mean)], [-k[1][golds_inx[k[0]]].item() for k in predictions_scores_mean]),4)
        rpp['PV'] = round(get_rpp([k[1][best_inx[i]].item() for i,k in enumerate(predictions_scores_var)], [-k[1][golds_inx[k[0]]].item() for k in predictions_scores_mean]),4)
        rpp['BALD'] = round(get_rpp([predictions_scores_bald[k[0]] for k in predictions_scores_mean],[-k[1][golds_inx[k[0]]].item() for k in predictions_scores_mean]),4)
    print("RPP: %s" % str(rpp))

    # to save top-1 for MC-dropout
    if n_Tsamples != 1:
        df = pd.DataFrame()
        df['instance_id'] = [k[0] for k in predictions_scores_mean]
        df['sense'] = [candidate_synsets[k[0]][best_inx[i]] for i,k in enumerate(predictions_scores_mean)]
        df['SMP_0'] = [round(1-k[1][best_inx[i]].item(),4) for i,k in enumerate(predictions_scores_mean)]
        df['PV_0'] = normalize(np.array([round(k[1][best_inx[i]].item(),4) for i,k in enumerate(predictions_scores_var)]))
        df['BALD_0'] = normalize(np.array([round(predictions_scores_bald[k[0]].item(),4) for k in predictions_scores_mean]))
        df['loss'] = [round(-k[1][golds_inx[k[0]]].item(),4) for k in predictions_scores_mean]
        df['Wrong_flag'] = [i for i in risk_binaries['MC']]
        for i in ['SMP', 'PV', 'BALD']:
            df['RCC_AUC_'+i] = [rcc_auc[i]] * len(df)
            df['RPP_'+i] = [rpp[i]] * len(df)
            df['CORR_'+i] = [df['loss'].corr(df[i+'_0'])] * len(df)
        # print(df.mean())
        print("The size of instances: %d"%len(df))
        print(df.mean())

        print(">>> Save to: " + "%s/%s_UE_prediction_MC_T%d_seed%d.csv"%(args.model_output_dir, args.mark, n_Tsamples,args.rand_seed))
        df.to_csv("%s/%s_UE_prediction_MC_T%d_seed%d.csv"%(args.model_output_dir, args.mark, n_Tsamples,args.rand_seed), index=False)

    # with open(args.model_output, 'w') as f:
    #     for instance_id, synset_id in predictions:
    #         f.write('{} {}\n'.format(instance_id, synset_id))
   
    # to save top 5 proper senses according to SR.
    else:
        df_SR = pd.DataFrame()
        df_SR['instance_id'] = [k[0] for k in predictions_topK]
        for i in range(1):
            df_SR["sense_"+str(i)] = [list(k[1][i].keys())[0] for k in predictions_topK]
            df_SR["MP_"+str(i)] = [round(1 - list(k[1][i].values())[0],4) for k in predictions_topK]
        df_SR['loss'] = [round(-k[1][golds_inx[k[0]]].item(),4) for k in predictions_scores]
        df_SR['Wrong_flag'] = [i for i in risk_binaries['SR']]
        df_SR['RCC_AUC_MP'] = [rcc_auc['MP']] * len(df_SR)
        df_SR['RPP_MP'] = [rpp['MP']] * len(df_SR)
        df_SR['CORR_MP'] = [(df_SR['loss'].corr(df_SR['MP_0']))] * len(df_SR)
        print("Save to: "+ "%s/%s_UE_prediction_SR_seed%d.csv"%(args.model_output_dir, args.mark, args.rand_seed))
        df_SR.to_csv("%s/%s_UE_prediction_SR_seed%d.csv"%(args.model_output_dir, args.mark, args.rand_seed), index=False)
        print("The size of instances: %d"%len(df_SR))
        print(df_SR.mean())
