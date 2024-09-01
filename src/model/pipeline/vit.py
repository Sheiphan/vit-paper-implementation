import torch
import torchvision
from torch import nn

import config
from loguru import logger

import matplotlib.pyplot as plt

print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    def __init__(self, in_channels, patch_size, embedding_dim):
        super().__init__()

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        self.flatten = nn.Flatten(
            start_dim=2,
            end_dim=3,
        )

    # 5. Define the forward method
    def forward(self, x):
        logger.warning("Entering PatchEmbedding")
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, logger.info(
            f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        )

        # Perform the forward pass
        x_patched = self.patcher(x)
        logger.info(f"Conv2D Output: {x.shape}")
        x_flattened = self.flatten(x_patched)
        logger.info(f"Flattened Output: {x.shape}")
        return x_flattened.permute(
            0, 2, 1
        )  # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, attn_dropout: int = 0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multuhead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multuhead_attn(query=x, key=x, value=x, need_weights=True)
        return attn_output


class MLP_block(nn.Module):
    def __init__(self, embedding_dim, mlp_size, dropout=0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TranformerEncoderBlock(nn.Module):
    def __init__(
        self, embedding_dim, num_heads, mlp_size, mlp_dropout=0.1, attn_dropout=0
    ):
        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )

        self.mlp_block = MLP_block(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    def forward(self, x):
        x = self.msa_block(x)
        x = self.mlp_block(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_channels=3,
        patch_size=16,
        num_transformer_layers=12,
        embedding_dim=768,
        mlp_size=3072,
        num_heads=12,
        attn_dropout=0,
        mlp_dropout=0.1,
        embedding_dropout=0.1,
        num_classes=10,
    ):
        super().__init__()

        self.img_height, self.img_width = img_size, img_size

        self.num_patch = (self.img_height * self.img_width) // patch_size**2

        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True
        )

        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patch + 1, embedding_dim)
        )

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        self.tranformer_encoder = nn.Sequential(
            *[
                TranformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    mlp_dropout=mlp_dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.classifer = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)

        logger.info(f"Patching embedding shape: {x.shape}")

        x = torch.cat((class_token, x), dim=1)

        logger.info(f"Patch embedding with class token shape: {x.shape}")

        x = self.position_embedding + x

        logger.info(f"Patch and position embedding shape: {x.shape}")

        x = self.embedding_dropout(x)

        x = self.tranformer_encoder(x)
        logger.info(f"Output Shape from Tranformer Encoder: {x.shape}")

        x = self.classifer(x[:, 0])
        logger.info(f"Ouput Shape from Classifier: {x.shape}")

        return x
