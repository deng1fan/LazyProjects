# copied from https://github.com/szprob/block/blob/a0afdc55dfdc7f6100f4d83d957cac509e5b8a42/src/block/models/bert/attention.py

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """Self attention for bert."""

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function of ScaledDotProductAttention.
        Args:
            q (torch.Tensor):
                Query. shape:(b,l,d) or (b ,h , l ,d )
            k (torch.Tensor):
                Key. shape:(b,l,d) or (b ,h , l ,d )
            v (torch.Tensor):
                Value. shape:(b,l,d) or (b ,h , l ,d )
            attn_mask (Optional[torch.Tensor], optional):
                Masking tensor. shape: (b, L_q, L_k)
                Defaults to None.
        Returns:
            torch.Tensor:
                shape : (b,l,d) or (b ,h , l ,d )
        """

        att = torch.matmul(q, k.transpose(-2, -1))

        # att:(b,l,l) or (b ,h , l ,l)
        att = att / math.sqrt(q.size(-1))

        # -np.inf or -1e9
        if attn_mask is not None:
            att = att.masked_fill_(attn_mask == 0, -1e9)
        att = F.softmax(att, dim=-1)

        # also can dropout here
        x = torch.matmul(att, v)

        # x : (b,l,d) or (b ,h , l ,d ) for multiheads
        # att: (b,l,l) or (b ,h , l ,l) for multiheads
        return x


class MultiHeadAttention(nn.Module):
    """Multi head self attention for bert.
    Attributes:
        d_model (int, optional):
            The dim of hidden layer.
            Defaults to 512.
        num_heads (int, optional):
            The number of attention heads.
            Defaults to 8.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        if not d_model % num_heads == 0:
            raise ValueError("""`d_model` should be divided by `num_heads`!""")

        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.fc_list = nn.ModuleList(
            [nn.Linear(d_model, d_model) for i in range(4)])
        self.attention = ScaledDotProductAttention()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function of ScaledDotProductAttention
        Args:
            q (torch.Tensor):
                Query. shape:(b,l,d)
            k (torch.Tensor):
                Key. shape:(b,l,d)
            v (torch.Tensor):
                Value. shape:(b,l,d)
            attn_mask (Optional[torch.Tensor], optional):
                Masking tensor. shape: (b, L_q, L_k)
                Defaults to None.
        Returns:
            torch.Tensor:
                shape : (b,l,d)
        """

        # q k v (b,l,d)
        # attn_mask (b,l,l)

        residual = q
        batch_size = k.size(0)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # attn_mask : [batch_size x n_heads x len_q x len_k]
        # linear projection split by heads
        q, k, v = [
            fc(x).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
            for fc, x in zip(self.fc_list, (q, k, v))
        ]
        # q k v (b,h,l,dk)
        # Apply attention on all the projected vectors in batch.
        x = self.attention(q, k, v, attn_mask=attn_mask)

        # x (b,h,l,dk)
        # attn (b,h,l,dk)
        # "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_head)
        )
        # x( b , l ,d )
        x = self.fc_list[-1](x)
        x = self.layer_norm(residual + x)
        return x

