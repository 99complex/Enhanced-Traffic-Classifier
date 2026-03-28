from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import Block, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed
import skimage
import skimage.filters.rank as sfr
from skimage.morphology import disk
import numpy as np

from modeling_etc_helper import LatentRegresser


# PatchEmbed 类是用于将图像分割成小块（patches），并将这些小块映射到一个高维空间的类。
class PatchEmbed(nn.Module):
    """ MTR matrix to Patch Embedding
    """
    def __init__(self, img_size=40, patch_size=2, in_chans=1, embed_dim=192):# 初始化函数，设置图像和补丁的尺寸、通道数和嵌入维度
        super().__init__()
        img_size = (int(img_size / 5), img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)# 创建一个2D卷积层，用于图像到嵌入的映射

#使用一个卷积层 self.proj 将图像分割成小块，并将每个小块映射到嵌入空间。然后，将结果展平并转置，以准备将其输入到 Transformer 模型中。
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# TrafficTransformer 类继承自 timm.models.vision_transformer.VisionTransformer 类，
# 并添加了自定义的 PatchEmbed 模块。它主要用于处理交通数据
class TrafficTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        # __init__方法在初始化时，除了调用父类的初始化方法外，还添加了一个自定义的PatchEmbed模块，并删除了原始的归一化层（del self.norm）
        #visiontransformer父类有位置嵌入函数
        super(TrafficTransformer, self).__init__(**kwargs)

        #  PatchEmbed 模块
        self.patch_embed = PatchEmbed(img_size=kwargs['img_size'], patch_size=kwargs['patch_size'],
                                         in_chans=kwargs['in_chans'], embed_dim=kwargs['embed_dim'])

        # 从关键字参数中获取归一化层的类型。
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)  # 使用获取的归一化层类型创建一个全连接层的归一化层。
        del self.norm  # remove the original norm 删除原始的归一化层，因为添加了一个新的归一化层。

    # 处理每个数据包的特征
    def forward_packet_features(self, x, i):
        # 获取批次大小
        B = x.shape[0]
        # 对输入 x 应用 PatchEmbed 模块
        x = self.patch_embed(x)

        # 扩展类别（class）标记以匹配批次大小
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        # 获取类别位置嵌入
        cls_pos = self.pos_embed[:, :1, :]
        # 根据索引 i 获取数据包的位置嵌入
        packet_pos = self.pos_embed[:, i*80+1:i*80+81, :]
        # 将类别位置嵌入和数据包位置嵌入连接起来
        pos_all = torch.cat((cls_pos, packet_pos), dim=1)
        # 将位置信息添加到嵌入的特征中
        x = x + pos_all
        # 对位置信息进行 dropout 处理
        x = self.pos_drop(x)

        # 遍历所有的 Transformer 块，并应用它们
        for blk in self.blocks:
            x = blk(x)

        # 提取类别标记的特征
        cls = x[:, :1, :]

        # 去除类别标记，保留数据包特征
        x = x[:, 1:, :]
        # 重塑数据并计算每个数据包的均值。
        x = x.reshape(B, 4, 20, -1).mean(axis=1)
        # 将类别特征和重塑后的数据包特征连接起来
        x = torch.cat((cls, x), dim=1)

        # 应用全连接层的归一化
        self.fc_norm(x)

        # 返回处理后的特征
        return x

    # 定义一个方法来处理整个输入特征
    def forward_features(self, x):
        # 解构输入 x 的形状，获取批次大小、通道数、高度和宽度。
        B, C, H, W = x.shape
        # 重塑输入以适应数据包处理
        x = x.reshape(B, C, 5, -1)
        # new_x = None #先定义一个变量
        # 遍历五个数据包。
        for i in range(5):
            # 选择当前数据包的特征。
            packet_x = x[:, :, i, :]
            # 进一步重塑数据包特征。
            packet_x = packet_x.reshape(B, C, -1, 40)
            # 对packet特征进行前向传播
            packet_x = self.forward_packet_features(packet_x, i)
            # 如果是第一个packet，直接赋值
            if i == 0:
                new_x = packet_x
            # 否则将当前packet特征与之前的结果在第1维度上进行拼接
            else:
                new_x = torch.cat((new_x, packet_x), dim=1)
        # 更新x为处理后的特征
        x = new_x

        # 对x应用blocks中的每个block
        for blk in self.blocks:
            x = blk(x)

        # 重塑x为(B, 5, 21, -1)的形状，并取每个序列的第0个元素
        x = x.reshape(B, 5, 21, -1)[:, :, 0, :]
        # 对x在第1维度上求平均
        x = x.mean(dim=1) #将 x 重塑为一个新形状的数组，其中每个批次包含5个数据包，每个数据包有21个特征，
        #对每个批次的5个数据包的第一个特征（class token）求平均。

        outcome = self.fc_norm(x)
        return outcome


class MaskedAutoencoder(nn.Module):
    """ Masked Autoencoder
    """
    """
          初始化函数，设置MaskedAutoencoder的参数。

          参数:
          img_size (int): 输入图像的大小。
          patch_size (int): 图像分割的大小。
          in_chans (int): 输入图像的通道数。
          embed_dim (int): 编码器的嵌入维度。
          depth (int): 编码器中Transformer块的数量。
          num_heads (int): 编码器中多头自注意力的头数。
          decoder_embed_dim (int): 解码器的嵌入维度。
          decoder_depth (int): 解码器中Transformer块的数量。
          decoder_num_heads (int): 解码器中多头自注意力的头数。
          mlp_ratio (float): MLP层的隐藏维度与输入维度之比。
          norm_layer (nn.Module): 使用的归一化层类型。
          norm_pix_loss (bool): 是否使用归一化的像素损失。
          """
    def __init__(self, img_size=40, patch_size=2, in_chans=1,
                 embed_dim=192, depth=4, num_heads=16,
                 decoder_embed_dim=192, decoder_depth=2, decoder_num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        #todo decoder_embed_dim=128  设为192试一下
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # 编码器部分的初始化，包括Patch Embedding和Transformer Blocks。
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches * 5

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #(1, 1, embed_dim)的全零张量
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        #(1, self.num_patches + 1, embed_dim)的全零张量

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        #加入对齐分支
        self.alignment_encoder =nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        # 如果编码器和回归器的嵌入维度不同，则添加一个线性层进行转换
        if decoder_embed_dim != embed_dim:
            self.encoder_to_regresser = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.encoder_to_regresser_norm = norm_layer(decoder_embed_dim)
        else:
            self.encoder_to_regresser = None
        # 加入回归器和解码器的位置嵌入
        self.rd_pos_embed=nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                      requires_grad=False)
        # self.regresser_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 初始化回归器
        self.regresser = LatentRegresser(embed_dim=decoder_embed_dim, regresser_depth=2,
                                         num_heads=decoder_num_heads,
                                         mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop_rate=0.,
                                         attn_drop_rate=0.,
                                         drop_path_rate=0., norm_layer=norm_layer,
                                         init_values=0.1, init_std=0.02)

        # 初始化遮罩标记
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # trunc_normal_(self.mask_token, std=self.init_std)
        # MAE decoder specifics
        # 加入回归器的位置嵌入
        self.rd_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                         requires_grad=False)

        # 如果解码器和回归器的嵌入维度不同，则添加一个线性层进行转换
        decoder_embed_dim2 = 128
        if decoder_embed_dim != decoder_embed_dim2:
            self.regresser_to_decoder = nn.Linear(decoder_embed_dim, decoder_embed_dim2, bias=True)
            self.regresser_to_decoder_norm = norm_layer(decoder_embed_dim2)
        else:
            self.regresser_to_decoder = None

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim2),
                                         requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim2, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim2)
        self.decoder_pred = nn.Linear(decoder_embed_dim2, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # 用主干网络的权重初始化对齐编码器
        self._init_alignment_encoder()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        """
            初始化权重。

            - 使用基于正弦-余弦的嵌入方法初始化（并冻结）pos_embed。
            - 使用基于正弦-余弦的嵌入方法初始化解码器的pos_embed。
            - 初始化patch_embed，使其类似于nn.Linear（而不是nn.Conv2d）。
            - 初始化cls_token和mask_token，使用标准差为0.02的正态分布。
            - 初始化所有nn.Linear和nn.LayerNorm层。
            """
        # 初始化（并冻结）pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化解码器的pos_embed
        rd_pos_embed = get_2d_sincos_pos_embed(self.rd_pos_embed.shape[-1],
                                                    int(self.num_patches ** .5), cls_token=True)
        self.rd_pos_embed.data.copy_(torch.from_numpy(rd_pos_embed).float().unsqueeze(0))
        # 初始化解码器的pos_embed
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                               int(self.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # 初始化patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # 初始化cls_token和mask_token
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        # 初始化所有nn.Linear和nn.LayerNorm层
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
           用于初始化nn.Linear和nn.LayerNorm层的辅助函数。

           参数:
           - m: 当前正在初始化的模块。
           """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        将图像分割成小块。
        参数:
        - imgs: 输入图像，形状为(N, 1, H, W)。
        返回:
        - x: 分割后的图像，形状为(N, L, patch_size**2 *1)。
        """

        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        将小块重新组合成图像。

        参数:
        - x: 分割后的图像，形状为(N, L, patch_size**2 *1)。

        返回:
        - imgs: 重新组合后的图像，形状为(N, 1, H, W)。
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        对输入序列进行随机遮掩。

        参数:
        - x: 输入序列，形状为(N, L, D)。
        - mask_ratio: 遮掩的比例。

        返回:
        - x_masked: 遮掩后的序列。
        - mask: 遮掩的二进制掩码。
        - ids_restore: 用于恢复原始顺序的索引。
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_del = ids_shuffle[:, len_keep:] #这行新代码计算了被掩码的元素的索引，即那些未被保留的元素。
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_masked_elements = torch.gather(x, dim=1, index=ids_del.unsqueeze(-1).repeat(1, 1, D))
        # ids_del 就包含了每个样本中被删除（掩码）的元素的索引，而 x_del 包含了这些被删除元素的实际值。
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # self.mask_token 是一个模型参数，它是一个形状为 (1, 1, mask_token_size) 的张量，其中 mask_token_size 是掩码标记的特征维度大小。
        # self.mask_token 来替换 x_del 中的所有元素。expand 函数用于扩展 self.mask_token 以匹配 x_del 的形状。-1 表示该维度保持原大小不变。
        # x_masked_elements = self.mask_token.expand(-1, x_masked_elements.size(1), -1)

        return x_masked, mask, ids_restore, x_masked_elements


    def _init_alignment_encoder(self):
        # 用主干网络的权重初始化对齐编码器
        # init the weights of alignment_encoder with those of backbone
        for param_encoder, param_alignment_encoder in zip(self.blocks.parameters(), self.alignment_encoder.parameters()):
            param_alignment_encoder.detach()
            param_alignment_encoder.data.copy_(param_encoder.data)
            param_alignment_encoder.requires_grad = False

    def alignment_parameter_update(self):
        """更新对齐编码器网络的参数."""
        """parameter update of the alignment_encoder network."""
        # 遍历基础编码器和对齐编码器的参数，将基础编码器的参数完全复制到对齐编码器中
        for param_encoder, param_alignment_encoder in zip(self.blocks.parameters(),
                                                self.alignment_encoder.parameters()):
            param_alignment_encoder.data = param_encoder.data # completely copy# 完全复制参数

    def forward_encoder(self, x, mask_ratio):
        """
            编码器的前向传播。

            参数:
            - x: 输入图像。
            - mask_ratio: 遮掩的比例。

            返回:
            - x: 编码器的输出。
            - mask: 遮掩的二进制掩码。
            - ids_restore: 用于恢复原始顺序的索引。
            """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, x_masked_elements = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x_masked_elements= self.norm(x_masked_elements)  #todo 9.24：尝试注释掉
        return x, mask, ids_restore, x_masked_elements #x_masked_elements掩码部分

    def forward_alignment_encoder(self,x_masked_elements):
        # 首先确保 alignment_encoder 的参数是最新的
        # 禁用梯度计算，使得前向传播不会更新权重
        with torch.no_grad():
            # 重复self.mask_token以匹配ids_restore的形状，用于后续的遮掩操作。

            # 应用 alignment_encoder 中的 Transformer 块进行前向传播 todo: 9.6  16.40添加class token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x_masked_elements.shape[0], -1, -1)
            latent_target=torch.cat((cls_tokens, x_masked_elements), dim=1)

            for blk in self.alignment_encoder:
                latent_target = blk(latent_target)

            latent_target = self.norm(latent_target)
            latent_target = latent_target[:, 1:, :] #remove class token #todo: 9.6  16.15添加
            # if self.encoder_to_regresser is not None: #todo:升维降维
            #     latent_target = self.encoder_to_regresser_norm(self.encoder_to_regresser(latent_target.detach()))

            self.alignment_parameter_update()

        # 返回预测的掩码特征
        return latent_target

    #todo:构造forward_decoder函数
    def forward_decoder(self, x_masked, pos_embed_masked):
        x_masked = x_masked + pos_embed_masked
        for blk in self.decoder_blocks:
            x_masked = blk(x_masked)
        x_masked = self.decoder_norm(x_masked)
        pred = self.decoder_pred(x_masked)
        # remove cls token  不需要移除cls token 掩码向量没有添加cls token
        # pred = x_masked[:, 1:, :]
        return pred

    def forward_loss(self, imgs, pred, mask, x_masked_predict, latent_target):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
            计算预测和目标之间的损失，仅在遮盖区域上计算。

            参数:
            imgs: 输入图像，形状为[N, 1, H, W]。
            pred: 模型对遮盖区域的预测，形状为[N, L, p*p*1]。
            mask: 遮盖区域的掩码，形状为[N, L]，0表示保留，1表示移除。

            返回:
            loss: 遮盖区域的平均损失。
        """
        # 将输入图像转换为patch表示
        target = self.patchify(imgs)
        # 如果使用像素标准化
        if self.norm_pix_loss:
            # 计算每个patch的均值和方差
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # 标准化patch
            target = (target - mean) / (var + 1.e-6) ** .5
        # print(target.shape) #torch.Size([256, 400, 4])
        # 确保mask是布尔型张量
        # print(mask.shape)  #torch.Size([256, 400])
        # print("掩码patch数量"+str(mask.sum(dim=1)))  #显示361
        mask = mask.bool()
        # 计算被掩码的 patch 数量
        num_masked = mask.sum(dim=1)
        # 初始化一个列表来存储每个样本的被掩码 patch
        target_masked_list = []
        for i in range(target.shape[0]):  # 遍历批次
            # 选择当前样本被掩码的 patch
            target_masked = target[i][mask[i]]
            # 将被掩码的 patch 添加到列表中
            target_masked_list.append(target_masked)
        # 将列表转换为张量
        target_masked = torch.stack(target_masked_list, dim=0)
        # print("target_masked shape:", target_masked.shape) #torch.Size([256, 361, 4])
        # print(pred.shape) #torch.Size([256, 361, 4])
        # 重建损失 计算预测和目标之间的平方差损失
        recon_loss = (pred - target_masked) ** 2  #（N，num_masked_patches，p^2*c）
        # 对每个patch的损失进行平均
        recon_loss = recon_loss.mean(dim=-1)  # [N, L], mean loss per patch
        # 对遮盖区域的损失进行加权求和，并计算平均损失
        recon_loss = recon_loss.sum()/mask.sum()

        # 计算对齐损失 (MSE损失)
        # align_loss = -latent_target * torch.log(x_masked_predict + 1e-8)  # 添加一个小的常数避免对数为负无穷
        # align_loss = align_loss.mean(dim=-1)  # [N, L], mean loss per patch
        # align_loss = align_loss.sum() / mask.sum()  # 总的平均损失
        align_loss = (x_masked_predict - latent_target) ** 2
        align_loss = align_loss.mean(dim=-1)  # [N, L], mean loss per patch
        align_loss = align_loss.sum() / mask.sum()  # 总的平均损失
        λ = 1
        loss = recon_loss + λ * align_loss
        return loss
    def forward(self, imgs, mask_ratio=0.60):
        """
           模型的前向传播函数。

           参数:
           imgs: 输入图像，形状为[N, 1, H, W]。
           mask_ratio: 遮盖区域的比例，默认为0.75。

           返回:
           loss: 遮盖区域的平均损失。
           pred: 模型对遮盖区域的预测，形状为[N, L, p*p*1]。
           mask: 遮盖区域的掩码，形状为[N, L]。
           """
        latent, mask, ids_restore, x_masked_elements = self.forward_encoder(imgs, mask_ratio)
        #对齐分支
        latent_target = self.forward_alignment_encoder(x_masked_elements)

        # todo :encoder到regresser的投影
        if self.encoder_to_regresser is not None:
            latent=self.encoder_to_regresser(latent)
            latent=self.encoder_to_regresser_norm(latent)  #暂时取名为latent1  后期改为latent

        # todo:准备掩码、未掩码的位置嵌入和掩码嵌入
        _, num_visible_plus1, dim = latent.shape

        #num_visible_plus1 表示除了类别标记外的补丁数量加一（类别标记），dim 是特征维度。

        x_cls_token = latent[:, :1, :]
        latent = latent[:, 1:, :] #移除类别标记

        batch_size=imgs.size(0)
        """
        以下为掩码、未掩码的位置嵌入和掩码嵌入
        """
        # ..........................................................................
        # 假设以下变量已经定义：
        # - batch_size: 批次大小
        # - self.num_patches: 总patch数量
        # - dim: 特征维度
        # - mask: [N, L], 0是保留，1是移除
        # - ids_restore: [N, L], 用于恢复原始顺序的索引

        # 扩展位置嵌入
        pos_embed = self.rd_pos_embed.expand(batch_size, self.num_patches + 1, dim).cuda(latent.device)
        pos_embed_decoder = self.decoder_pos_embed.expand(batch_size, self.num_patches + 1, 128).cuda(latent.device)
        # 初始化掩码和未掩码的位置嵌入张量
        pos_embed_masked = torch.zeros([batch_size, self.num_patches, dim])
        pos_embed_unmasked = torch.zeros([batch_size, self.num_patches, dim])
        pos_embed_masked_decoder = torch.zeros([batch_size, self.num_patches, 128])
        pos_embed_unmasked_decoder = torch.zeros([batch_size, self.num_patches, 128])
        # 使用布尔索引来分配遮掩和未遮掩的位置嵌入
        bool_masked_pos = (mask == 1).unsqueeze(-1).expand(-1, -1, dim)  # 扩展mask以匹配pos_embed的形状
        bool_masked_pos_decoder = (mask == 1).unsqueeze(-1).expand(-1, -1, 128)  # 扩展mask以匹配pos_embed的形状
        bool_unmasked_pos = (mask == 0).unsqueeze(-1).expand(-1, -1, dim)
        # 被掩码补丁的位置嵌入
        pos_embed_masked = pos_embed[:, 1:][bool_masked_pos].reshape(batch_size, -1, dim)
        pos_embed_masked_decoder = pos_embed_decoder[:, 1:][bool_masked_pos_decoder].reshape(batch_size, -1, 128)
        # 未掩码补丁的位置嵌入
        pos_embed_unmasked = pos_embed[:, 1:][bool_unmasked_pos].reshape(batch_size, -1, dim)

        # 初始化x_masked
        num_masked_patches=self.num_patches-(num_visible_plus1-1)
        x_masked = self.mask_token.expand(batch_size, num_masked_patches, -1) #mask掩码嵌入
        # print("num_masked_patches", num_masked_patches)

        # .........................................................................
        # 2. 通过回归器预测掩码的潜在变量 #todo 测试连通性  这里的latent1 后面改为latent
        x_masked_predict = self.regresser(x_masked, latent, pos_embed_masked,
                                          pos_embed_unmasked)  # 2. 通过回归器预测掩码的潜在变量
        # print("x_masked_predict.shape",x_masked_predict.shape) #torch.Size([256, 361, 128])
        x_masked_predict_dim192 = x_masked_predict
        # todo :encoder到regresser的投影
        if self.regresser_to_decoder is not None:
            x_masked_predict = self.regresser_to_decoder(x_masked_predict)
            x_masked_predict = self.regresser_to_decoder_norm(x_masked_predict)  # 暂时取名为latent1  后期改为latent
        pred = self.forward_decoder(x_masked_predict, pos_embed_masked_decoder)
        # print("pred.shape", pred.shape) #([256, 361, 4])
        loss = self.forward_loss(imgs, pred, mask, x_masked_predict_dim192, latent_target)
        return loss, pred, x_masked_predict, mask


# for pre-training
# 预训练模型定义
def MAE_YaTC(**kwargs):
    """
    创建一个MaskedAutoencoder模型实例，用于预训练。
    参数:
    **kwargs: 传递给MaskedAutoencoder构造函数的额外参数。
    返回:
    model: 创建的MaskedAutoencoder模型实例。
    """
    model = MaskedAutoencoder(
        img_size=40, patch_size=2, embed_dim=192, depth=4, num_heads=16,
        decoder_embed_dim=192, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer = partial(nn.LayerNorm, eps=1e-6) , **kwargs)
    #todo  decoder_embed_dim=128（原来的值）
    return model


# for fine-tuning
# 微调调模型定义
def TraFormer_YaTC(**kwargs):
    """
    创建一个TrafficTransformer模型实例，用于精调。
    参数:
    **kwargs: 传递给TrafficTransformer构造函数的额外参数。
    返回:
    model: 创建的TrafficTransformer模型实例。
    """
    model = TrafficTransformer(
        img_size=40, patch_size=2, in_chans=1, embed_dim=192, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

