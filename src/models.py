import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import timm
from timm.layers import LayerNorm2d
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

class CustomSwinTransformerModel(nn.Module):
    def __init__(self, model_name, num_classes: int = 1854, pretrained: bool = True,
                 aux_loss_ratio: float = None, dropout_rate: float = 0.05):
        super(CustomSwinTransformerModel, self).__init__()
        self.aux_loss_ratio = aux_loss_ratio
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        self.features = nn.Sequential(*list(self.encoder.children())[:-1])
        self.GAP = SelectAdaptivePool2d(pool_type='avg', input_fmt='NHWC',flatten=True)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder.num_features, num_classes)
        )
        if aux_loss_ratio is not None:
            self.decoder_aux = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.encoder.num_features, 4)
            )

    def expand_dims(self, images):
        images = images.unsqueeze(1).expand(-1, 3, -1, -1)
        return images

    def forward(self, images):
        images = self.expand_dims(images)
        out = self.features(images)
        out = self.GAP(out)
        main_out = self.decoder(out)

        if self.aux_loss_ratio is not None:
            out_aux = self.decoder_aux(out)
            return main_out, out_aux
        else:
            return main_out
        


class CustomConvNextModel(nn.Module):
    def __init__(self, model_name, num_classes: int = 1854, pretrained: bool = True, 
                 aux_loss_ratio: float = None, dropout_rate: float = 0.05):
        super(CustomConvNextModel, self).__init__()
        self.aux_loss_ratio = aux_loss_ratio
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.layer_norm = LayerNorm2d(self.encoder.num_features)
        self.flatten = nn.Flatten()
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder.num_features, num_classes)
        )
        if aux_loss_ratio is not None:
            self.decoder_aux = nn.Sequential(
  
                nn.Dropout(dropout_rate),
                nn.Linear(self.encoder.num_features, 4)
            )
        
    def expand_dims(self, images):
        images = images.unsqueeze(1).expand(-1, 3, -1, -1)
        return images

    def forward(self, images):
        images = self.expand_dims(images)
        out = self.features(images)
        out = self.GAP(out)
        out = self.layer_norm(out)
        out = self.flatten(out)
        main_out = self.decoder(out)
        
        if self.aux_loss_ratio is not None:
            out_aux = self.decoder_aux(out)
            return main_out, out_aux
        else:
            return main_out


class CustomEfficientNetModel(nn.Module):
    def __init__(self, model_name, num_classes: int = 1854, pretrained: bool = True, 
                  aux_loss_ratio: float = None, dropout_rate: float = 0):
        super(CustomEfficientNetModel, self).__init__()
        self.aux_loss_ratio = aux_loss_ratio
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder.num_features, num_classes)
        )
        if aux_loss_ratio is not None:
            self.decoder_aux = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.encoder.num_features, 4)
                )
        
    def expand_dims(self, images):
        # Expand dims to [B, H, W, 3]
        images = images.unsqueeze(1).expand(-1, 3, -1, -1)
        return images

    def forward(self, images):
        images = self.expand_dims(images)
        out = self.features(images)
        out = self.GAP(out)
        main_out = self.decoder(out.view(out.size(0), -1))
        
        if self.aux_loss_ratio is not None:
            out_aux = self.decoder_aux(out.view(out.size(0), -1))
            return main_out, out_aux
        else:
            return main_out


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)