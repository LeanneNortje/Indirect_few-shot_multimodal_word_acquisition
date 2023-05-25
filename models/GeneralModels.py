#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoringAttentionModule(nn.Module):
    def __init__(self, args):
        super(ScoringAttentionModule, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.image_attention_encoder = nn.Sequential(
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.audio_attention_encoder = nn.Sequential(
            # nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.pool_func = nn.AdaptiveAvgPool2d((1, 1))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, image_embedding, audio_embeddings, audio_nframes):

        # im = self.pool_func(image_embedding.transpose(1, 2).unsqueeze(2)).squeeze(-1).transpose(1, 2)
        # aud = self.pool_func(audio_embeddings.unsqueeze(2)).squeeze(-1).transpose(1, 2)
        # im = self.image_embedding_encoder(im)
        # aud = self.audio_embedding_encoder(aud)
        att = torch.bmm(image_embedding, audio_embeddings)
        att, _ = att.max(dim=1)
        s = att.mean(dim=-1)

        return s

    def encode(self, image_embedding, audio_embeddings, audio_nframes):

        att = torch.bmm(image_embedding, audio_embeddings)
        att, _ = att.max(dim=1)
        s = att.mean(dim=-1)
        # im = self.pool_func(image_embedding.transpose(1, 2).unsqueeze(2)).squeeze(-1).transpose(1, 2).squeeze(1)
        # aud = self.pool_func(audio_embeddings.unsqueeze(2)).squeeze(-1).transpose(1, 2).squeeze(1)
        # s = self.cos(im, aud)

        return s, att.unsqueeze(1)

    def one_to_many_score_audio(self, audio_sup, audio_embeddings, audio_nframes):
        
        scores = [] #torch.zeros((audio_embeddings.size(0), image_embedding.size(0)), device=audio_embeddings.device)
        for i in range(audio_sup.size(0)):
            im = audio_sup[i, :, :].unsqueeze(0)
            att = torch.bmm(audio_embeddings.transpose(1, 2), im)# / (torch.norm(aud_em, dim=1) * torch.norm(im, dim=1))
            att, _ = att.max(dim=-1)
            s, _ = att.max(dim=-1)
            # print(s.size())
            scores.append(s.unsqueeze(1))
        scores = torch.cat(scores, dim=1)
        return scores #self.sig(scores)

    def one_to_many_score_images(self, image_sup, image_embedding):
        
        scores = [] #torch.zeros((audio_embeddings.size(0), image_embedding.size(0)), device=audio_embeddings.device)
        for i in range(image_sup.size(0)):
            im = image_sup[i, :, :].unsqueeze(0).transpose(1, 2)
            att = torch.bmm(image_embedding, im)# / (torch.norm(aud_em, dim=1) * torch.norm(im, dim=1))
            
            att = att.mean(dim=-1)
            s = att.mean(dim=-1)
            # print(s.size())
            scores.append(s.unsqueeze(1))
        scores = torch.cat(scores, dim=1)
        return scores #self.sig(scores)

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.margin = args["margin"]
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, anchor, positives, negatives):

        for p in range(positives.size(1)):
            samples = torch.cat([positives[:, p, :].unsqueeze(1), negatives], dim=1)
            sim = []
            for i in range(anchor.size(0)):
                sim.append(self.cos(anchor[i, :, :].repeat(samples.size(1), 1), samples[i, :, :]).unsqueeze(0))
            sim = torch.cat(sim, dim=0)
            labels = torch.zeros(sim.size(0), dtype=torch.long, device=sim.device)
            loss = F.cross_entropy(sim, labels)

        return loss

    def encode(self, anchor, positives, negatives):

        samples = torch.cat([positives, negatives], dim=1)
        # sim = torch.bmm(anchor, samples.transpose(1, 2)).squeeze(1)
        sim = []
        for i in range(anchor.size(0)):
            sim.append(self.cos(anchor[i, :, :].repeat(samples.size(1), 1), samples[i, :, :]).unsqueeze(0))
        sim = torch.cat(sim, dim=0)
        labels = torch.zeros(sim.size(0), dtype=torch.long, device=sim.device)

        return sim, labels