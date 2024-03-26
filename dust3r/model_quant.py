# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import torch.nn as nn

from .utils.misc import fill_default_args, freeze_all_params, interleave
from .heads import head_factory
from dust3r.patch_embed import PatchEmbedDust3RQuant

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa
from models.blocks import Block
inf = float('inf')

def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}

def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        # B = len(true_shape)
        # assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        # B = len(true_shape)
        B = true_shape.size(0)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait),  (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no

class RoPE2DQuant(torch.nn.Module):
        
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, tokens, positions, device, dtype):
        D = tokens.size(3) // 2
        seq_len = positions.max().int()+1
        if (D,seq_len,device,dtype) not in self.cache:
            # inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            inv_freq = 1.0 / (self.base ** ((tokens.new_ones((D + 1) // 2) * 2).cumsum(0) - 2) / D)
            # t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            t = inv_freq.new_ones(seq_len).cumsum(0) - 1
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos() # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D,seq_len,device,dtype] = (cos,sin)
        return self.cache[D,seq_len,device,dtype]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        pos1d = pos1d.long()
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)
        
    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        # assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
        # D = tokens.size(3) // 2
        # assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
        cos, sin = self.get_cos_sin(tokens, positions, tokens.device, tokens.dtype)
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
        x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens


class AsymmetricCroCo3DStereoQuant(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        self.rope = RoPE2DQuant(freq=float(croco_kwargs['pos_embed'][4:]))
        self.enc_blocks = nn.ModuleList([
            Block(croco_kwargs['enc_embed_dim'], croco_kwargs['enc_num_heads'], croco_kwargs['mlp_ratio'], \
                  qkv_bias=True, norm_layer=croco_kwargs['norm_layer'], rope=self.rope)
            for i in range(croco_kwargs['enc_depth'])])
        
        self._set_decoder(croco_kwargs['enc_embed_dim'], croco_kwargs['dec_embed_dim'], \
                          croco_kwargs['dec_num_heads'], croco_kwargs['dec_depth'], \
                          croco_kwargs['mlp_ratio'], croco_kwargs['norm_layer'], \
                          croco_kwargs['norm_im2_in_dec'])
        self.dec_blocks2 = deepcopy(self.dec_blocks)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbedDust3RQuant(img_size, patch_size, 3, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        # if img1.shape[-2:] == img2.shape[-2:]:
        out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                            torch.cat((true_shape1, true_shape2), dim=0))
        out, out2 = out.chunk(2, dim=0)
        pos, pos2 = pos.chunk(2, dim=0)
        # else:
        #     out, pos, _ = self._encode_image(img1, true_shape1)
        #     out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2
    
    def is_symmetrized(self, x, y):
        if len(x) == len(y) and len(x) == 1:
            return False  # special case of batchsize 1
        ok = True
        for i in range(0, len(x), 2):
            ok = ok and (x[i] == y[i+1]) and (x[i+1] == y[i])
        return ok

    def _encode_symmetrized(self, img1, img2, shape1, shape2, instance1, instance2):
        B = img1.shape[0]
        # warning! maybe the images have different portrait/landscape orientations

        # if self.is_symmetrized(instance1, instance2):
        #     # computing half of forward pass!'
        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
        feat1, feat2 = interleave(feat1, feat2)
        pos1, pos2 = interleave(pos1, pos2)
        # else:
        #     feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, img1, img2, shape1, shape2, instance1, instance2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(img1, img2, shape1, shape2, instance1, instance2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
