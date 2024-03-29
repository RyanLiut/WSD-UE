import torch
import torch.nn as nn
import torch_scatter as scatter
from transformers import AutoModel, AutoConfig


class WordEncoder(nn.Module):

    def __init__(self, hparams, padding_token_id):
        super(WordEncoder, self).__init__()

        self.word_embedding = BertEmbedding(
            model_name=hparams.language_model,
            fine_tune=hparams.language_model_fine_tuning)

        if 'base' in hparams.language_model:
            word_embedding_size = 768
        else:
            word_embedding_size = 1024

        self.batch_normalization = nn.BatchNorm1d(word_embedding_size)
        self.projection = nn.Linear(word_embedding_size, hparams.word_projection_size, bias=False)
        self.word_dropout = nn.Dropout(hparams.word_dropout)

        self.word_embedding_size = hparams.word_projection_size

    def forward(self, word_ids, subword_indices=None, sequence_lengths=None):
        word_embeddings, attention = self.word_embedding(word_ids, sequence_lengths=sequence_lengths)
        word_embeddings = scatter.scatter_mean(word_embeddings, subword_indices, dim=1)
        word_embeddings = word_embeddings[:, 1:, :]
        word_embeddings = self.word_dropout(word_embeddings)
        word_embeddings = word_embeddings.permute(0, 2, 1)
        word_embeddings = self.batch_normalization(word_embeddings)
        word_embeddings = word_embeddings.permute(0, 2, 1)
        word_embeddings = self.projection(word_embeddings)
        word_embeddings = word_embeddings * torch.sigmoid(word_embeddings)
        word_embeddings = self.word_dropout(word_embeddings)
        return word_embeddings, attention


class BertEmbedding(nn.Module):

    def __init__(self, model_name='bert-base-multilingual-cased', fine_tune=False):
        super(BertEmbedding, self).__init__()
        self.fine_tune = fine_tune
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        if not fine_tune:
            self.bert.eval()

    def forward(self, word_ids, subword_indices=None, sequence_lengths=None):
        timesteps = word_ids.shape[1]
        attention_mask = torch.arange(timesteps, device='cuda' if torch.cuda.is_available() else 'cpu').unsqueeze(0) < sequence_lengths.unsqueeze(1)

        if not self.fine_tune:
            with torch.no_grad():
                word_embeddings = self.bert(
                    input_ids=word_ids,
                    attention_mask=attention_mask)
        else:
            word_embeddings = self.bert(
                input_ids=word_ids,
                attention_mask=attention_mask)

        word_embeddings_new = 0.25 * (word_embeddings[2][-1] + word_embeddings[2][-2] + word_embeddings[2][-3] + word_embeddings[2][-4])
        # To visualize, output the attention weights.
        attention = word_embeddings[-1]
        
        return word_embeddings_new, attention
