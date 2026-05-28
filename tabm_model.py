from dataclasses import dataclass, field
from pytorch_tabular.config import ModelConfig
from pytorch_tabular.models import BaseModel
from pytorch_tabular.models.common.layers import Embedding1dLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TabMConfig(ModelConfig):
    k: int = 5 # Number of ensemble members
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    use_batch_norm: bool = False # BatchEnsemble interacts poorly with BatchNorm, use LayerNorm or skip it

class BatchEnsembleLinear(nn.Module):
    def __init__(self, in_features, out_features, k):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.r = nn.Parameter(torch.empty(k, in_features))
        self.s = nn.Parameter(torch.empty(k, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features))
        
        nn.init.normal_(self.r, mean=1.0, std=0.1)
        nn.init.normal_(self.s, mean=1.0, std=0.1)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: (B, in_features) or (B, K, in_features)
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.k, -1)
            
        x_scaled = x * self.r
        out = self.linear(x_scaled)
        out = out * self.s + self.bias
        return out

class TabMBackbone(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        inp_dim = hparams.embedded_cat_dim + hparams.continuous_dim
        
        layers = []
        for i in range(self.hparams.num_layers):
            layers.append(BatchEnsembleLinear(inp_dim, self.hparams.hidden_dim, self.hparams.k))
            if self.hparams.use_batch_norm:
                layers.append(nn.LayerNorm(self.hparams.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hparams.dropout))
            inp_dim = self.hparams.hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.output_dim = self.hparams.hidden_dim
        
    def forward(self, x: torch.Tensor):
        return self.network(x)
        
    def _build_embedding_layer(self):
        return Embedding1dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_embedding_dims=self.hparams.embedding_dims,
            embedding_dropout=self.hparams.embedding_dropout,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
        )

class TabMModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        self._backbone = TabMBackbone(self.hparams)
        self._embedding_layer = self._backbone._build_embedding_layer()
        self._head = self._get_head_from_config()

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head

    def compute_head(self, backbone_features: torch.Tensor):
        # backbone_features: (B, K, out_features)
        y_hat = self.head(backbone_features) # (B, K, num_classes)
        y_hat = y_hat.mean(dim=1) # (B, num_classes) -> average predictions across ensemble
        y_hat = self.apply_output_sigmoid_scaling(y_hat)
        return self.pack_output(y_hat, backbone_features.mean(dim=1))
