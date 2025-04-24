import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import layers
import commons
import modules

from modules import PositionalEncodingComponent, FeedForwardComponent, MultiHeadedAttentionComponent, EncodingLayer
from modules2 import TransformerBlock, FeedForwardBlock, TransformerEncoderBlock, CrossmodalTransformer

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TDNNLayer(nn.Module):
    def __init__(self, tdnn_dim, tdnn_kernel, tdnn_dilation, layer_id=0):
        super().__init__()
        self.in_conv_dim = tdnn_dim[layer_id - 1] if layer_id > 0 else tdnn_dim[layer_id]
        self.out_conv_dim = tdnn_dim[layer_id]
        self.kernel_size = tdnn_kernel[layer_id]
        self.dilation = tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.activation(hidden_states)
        return hidden_states

class DimensionPredictorVITS(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, out_channel=1, gin_channels=0, txtin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    self.txtin_channels = txtin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, out_channel, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    if txtin_channels != 0:
      self.cond_txt = nn.Conv1d(txtin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None, txt=None):
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

    if txt is not None:
      txt = torch.detach(txt)
      x = x + self.cond_txt(txt)

    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

class TransformerLayer(nn.Module):
    def __init__(self, d_dim, d_dim2, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_dim, kdim=d_dim2, vdim=d_dim2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_dim)
        )
        self.norm3 = nn.LayerNorm(d_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, key, value):
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-Attention
        cross_attn_out, _ = self.cross_attn(x, key, value)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))   
        return x

class AttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their weighted average as output. These weights comes from an attention mechanism.
        It can be seen as a One Head Self-Attention, where a unique query is used and input vectors are the values and keys.   
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()

        self.emb_in = emb_in
        self.init_query()
        
    def init_query(self):
        self.query = torch.nn.Parameter(torch.FloatTensor(self.emb_in, 1))
        torch.nn.init.xavier_normal_(self.query)


    def forward(self, input_tensors):
        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'
        attention_scores = torch.matmul(input_tensors, self.query)
        attention_scores = attention_scores.squeeze(dim = -1)
        attention_scores = F.softmax(attention_scores, dim = 1)
        attention_scores = attention_scores.unsqueeze(dim = -1)
        output = torch.bmm(attention_scores.transpose(1, 2), input_tensors)
        output = output.view(output.size()[0], output.size()[1] * output.size()[2])
        
        return output

class CrossTransformerLayer(nn.Module):
    def __init__(self, dim_speech, dim_text, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_speech, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_speech, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=dim_speech, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(dim_speech, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim_speech)
        )
        
        self.norm1 = nn.LayerNorm(dim_speech)
        self.norm2 = nn.LayerNorm(dim_speech)
        self.norm3 = nn.LayerNorm(dim_speech)
        self.norm4 = nn.LayerNorm(dim_speech)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, speech, speech_mask, text, text_mask):
        # Create padding masks
        speech_padding_mask = self.create_padding_mask(speech_mask, speech.shape[1])
        text_padding_mask = self.create_padding_mask(text_mask, text.shape[1])
        
        # Self-Attention on Speech
        attn_speech, _ = self.self_attn(speech, speech, speech, key_padding_mask=speech_padding_mask)
        speech = self.norm1(speech + self.dropout(attn_speech))

        # Cross-Attention2 (Speech Query, Text Key/Value)
        attn_text_speech, _ = self.cross_attn2(text, speech, speech, key_padding_mask=speech_padding_mask)
        text = self.norm2(text + self.dropout(attn_text_speech))
        
        # Cross-Attention (Speech Query, Text Key/Value)
        attn_speech_text, _ = self.cross_attn(speech, text, text, key_padding_mask=text_padding_mask)
        speech = self.norm3(speech + self.dropout(attn_speech_text))
        
        # Feedforward
        speech_ff = self.ff(speech)
        speech = self.norm4(speech + self.dropout(speech_ff))
        
        return speech

    def create_padding_mask(self, lengths, max_len):
        """
        Create a padding mask from lengths.
        True for padded positions, False otherwise.
        """
        batch_size = len(lengths)
        mask = torch.arange(max_len, dtype=lengths.dtype, device=lengths.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask

class AttentiveStatisticsPooling(nn.Module):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs, feat_lens):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim*2)
        """
        pooled_list = []
        for x, feat_len in zip(xs, feat_lens):
            x = x[:feat_len].unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)


class TextRepresentations(nn.Module):
    """
    Group of layers that give final text representation for cross attention
    """

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, text_pad_index, max_length=5000):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, max_length)

        # encoder layers
        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

        self.text_pad_index = text_pad_index

    def create_text_mask(self, text):
        # masks padded values of text

        # text : [batch_size, src_len]
        text_mask = (text != self.text_pad_index).unsqueeze(1).unsqueeze(2)

        return text_mask

    def forward(self, text):
        # text : [batch_size, src_len]

        text_mask = self.create_text_mask(text)
        # text_mask : [batch_size,1,1,src_len]

        batch_size = text.shape[0]
        src_len = text.shape[1]

        tok_embeddings = self.tok_embedding(text) * self.scale

        # token plus position embeddings
        text = self.pos_embedding(tok_embeddings)

        for layer in self.layers:
            text = layer(text, text_mask)
        # src : [batch_size, src_len, hid_dim]

        return text


# Cross Attention Layer
class CrossAttentionLayer(nn.Module):
    '''
    This layer takes input the audio and text representations after they have been 
    passed through their respective Encoding layers. 
    The text representations will act as query
    the audio representations will be key and values.
    So this will take most important features from text representation based on the
    attention between audio and the text features.
    '''

    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after self-attention
        self.ff_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after FeedForward component

        self.self_attention = MultiHeadedAttentionComponent(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio):
        # text : [batch_size, text_len, hid_dim]
        # audio : [batch_size, audio_len, hid_dim

        # get self-attention
        _text, _ = self.self_attention(text, audio, audio)

        # LayerNorm after dropout
        text = self.self_attn_layer_norm(text + self.dropout(_text))
        # text : [batch_size, text_len, hid_dim]

        # FeedForward
        _text = self.feed_forward(text)

        # layerNorm after dropout
        text = self.ff_layer_norm(text + self.dropout(_text))
        # text: [batch_size, text_len, hid_dim]

        return text

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

class SimplePooling(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", **kwargs):
        super(SimplePooling, self).__init__()
        
        self.eps = 1e-5
        self.temporal_pred = temporal_pred

        # Learnable scale factors
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

        # if self.temporal_pred == "GlowDP":
        #     self.dim_pred = DimensionPredictor(ssl_hidden_channels, filter_channels, kernel_size, p_dropout)
        # elif self.temporal_pred == "GlowDP_spkembed":
        #     self.dim_pred = DimensionPredictorVITS(ssl_hidden_channels, filter_channels, kernel_size, 
        #                                             p_dropout, out_channel=d_dim, 
        #                                             gin_channels=spk_embed_dim, txtin_channels=txt_embed_dim)

        self.layer_norm = nn.LayerNorm(ssl_hidden_channels)

        self.sof = nn.Linear(ssl_hidden_channels, 1)  # Attention weights
        self.lin = nn.Linear(ssl_hidden_channels, ssl_hidden_channels)  # Feature transformation

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.

        Arguments
        ---------
        shape_of_tensor : torch.Tensor
            It represents the size of tensor for generating Gaussian noise.
        device : str
            Device on which to perform computations.

        Returns
        -------
        gnoise : torch.Tensor
            The Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None): # x [b, 2028]
        #x = x.transpose(1, 2)                   # [B, hidden_channels, T]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.transpose(1, 2).size(2)), 1).to(x.dtype) # [B, 1, T] 

        spk_embeds = self.emb_g(F.normalize(spk_embeds))
        txt_embeds = self.emb_txt(F.normalize(txt_embeds))

        # mean_features = []
        # for i, length in enumerate(x_lengths):
        #     length = int(length)
        #     mean_features.append(x[i, :length].mean(dim=0))
        # x_pooled = torch.stack(mean_features)
        # gnoise = self._get_gauss_noise(x_pooled.size(), device=x_pooled.device)
        # x_pooled += gnoise

        attn_scores = self.sof(x).squeeze(-1)  # (Batch, Time)

        if x_mask is not None:
            attn_scores = attn_scores.masked_fill(x_mask.squeeze(1) == 0, float('-inf'))  # Mask out padding before softmax

        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (Batch, Time, 1)
        transformed_x = self.lin(x)  # (Batch, Time, Hidden)
        x_pooled = torch.sum(attn_weights * transformed_x, dim=1)  # (Batch, Hidden)  # (Batch, Hidden)
        
        # x = torch.cat((x, l.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)
        decoder_out = self.alpha * x_pooled + self.beta * spk_embeds + self.gamma * txt_embeds
        decoder_out = self.layer_norm(decoder_out)

        # spk_embeds = spk_embeds.unsqueeze(-1)   # [B, spk_embed_dim, 1]
        # txt_embeds = txt_embeds.unsqueeze(-1)   # [B, spk_embed_dim, 1]
        # x = x.transpose(1, 2)                   # [B, hidden_channels, T]
        # x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype) # [B, 1, T] 

        # hidden_states = self.dim_pred(x, x_mask, g=spk_embeds, txt=txt_embeds) # [B, pooling_hidden, T] 
        # hidden_states = hidden_states.permute(0, 2, 1) # [B, T, H] 

        # mean_features = []
        # std_features = []
        # for i, length in enumerate(x_lengths):
        #     length = int(length)
        #     mean_features.append(hidden_states[i, :length].mean(dim=0))
        #     std_features.append(hidden_states[i, :length].std(dim=0))
        # mean_features = torch.stack(mean_features)
        # std_features = torch.stack(std_features)
        # statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        # decoder_out = self.projection_out(statistic_pooling)
        return decoder_out


class CrossPooling(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(CrossPooling, self).__init__()
        
        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        self.emb_g = nn.Linear(spk_embed_dim, ssl_hidden_channels)

        self.layer_norm_x = nn.LayerNorm(ssl_hidden_channels)
        self.layer_norm_txt = nn.LayerNorm(ssl_hidden_channels)

        num_layers = 4
        self.layers = nn.ModuleList([
            CrossTransformerLayer(ssl_hidden_channels, txt_embed_dim, 8, ssl_hidden_channels * 2, 0.1) 
            for _ in range(num_layers)
        ])

        self.x_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
        txt_masks = torch.sum(txt_masks, dim=1).long()
        x_mask = commons.sequence_mask(x_lengths, x.size(1)).to(x.dtype).unsqueeze(-1) # [B, T, 1] 

        spk_embeds = self.emb_g(F.normalize(spk_embeds)).unsqueeze(1) # torch.Size([11, 1, 1024])
        spk_embeds = spk_embeds.expand(x.size(0), x.size(1), x.size(2))
        hidden_states = x + spk_embeds
        hidden_states = hidden_states * x_mask
        
        txt_embeds = self.layer_norm_txt(txt_embeds)
        hidden_states = self.layer_norm_x(hidden_states)
       
        # hidden_states = torch.cat([spk_embeds, x], dim=1)
        # x_lengths = x_lengths + 1

        for layer in self.layers:
            hidden_states = layer(hidden_states, x_lengths, txt_embeds, txt_masks)

        decoder_out = self.x_pooling(hidden_states, x_lengths)
        return decoder_out

class SimplePooling2(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", **kwargs):
        super(SimplePooling2, self).__init__()
        
        self.eps = 1e-5
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        self.layer_norm_x = nn.LayerNorm(ssl_hidden_channels)
        self.x_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)

    def forward(self, x, x_lengths, attention_mask=None): # x [b, 2028] or [11, 25, 406, 1024])
        hidden_states = x
        hidden_states = self.layer_norm_x(hidden_states)

        x_pooled = self.x_pooling(hidden_states, x_lengths)

        decoder_out = x_pooled #torch.cat([x_pooled, txt_pooled], dim=1)
        return decoder_out

class TransDecoderPooling(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(TransDecoderPooling, self).__init__()

        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred

        if self.temporal_pred == "GlowDP":
            self.dim_pred = DimensionPredictor(ssl_hidden_channels, filter_channels, kernel_size, p_dropout)
        elif self.temporal_pred == "GlowDP_spkembed":
            self.dim_pred = DimensionPredictorVITS(ssl_hidden_channels, filter_channels, kernel_size, 
                                                    p_dropout, out_channel=d_dim, 
                                                    gin_channels=spk_embed_dim, txtin_channels=txt_embed_dim)

        self.bos_idx = 0
        embedding_pos = nn.Embedding(num_embeddings=1, embedding_dim=d_dim, padding_idx=0)
        pos_encoder = SinusoidalPositionalEncoding(d_dim)
        self.frontend_decoder = nn.Sequential(embedding_pos, pos_encoder)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_dim, nhead=n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.projection_out = nn.Linear(d_dim, pooling_hidden)

    def _get_pooling_tokens(self, batch_size, device):
        """Generate the initial pooling tokens (BOS tokens)."""
        return torch.full((1, batch_size), self.bos_idx, dtype=torch.long, device=device)

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None): # x [b, 2028]
        spk_embeds = spk_embeds.unsqueeze(-1)   # [B, spk_embed_dim, 1]
        txt_embeds = txt_embeds.unsqueeze(-1)   # [B, spk_embed_dim, 1]
        x = x.transpose(1, 2)                   # [B, hidden_channels, T]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype) # [B, 1, T] 

        dim_proj = self.dim_pred(x, x_mask, g=spk_embeds, txt=txt_embeds) # [B, pooling_hidden, T] 
        dim_proj = dim_proj.permute(2, 0, 1) # [T, B, H] 
        #dim_proj = x.permute(2, 0, 1)

        pooling_tokens = self._get_pooling_tokens(dim_proj.size(1), dim_proj.device)
        pooling_tokens = self.frontend_decoder(pooling_tokens)
        decoder_output = self.decoder(pooling_tokens, dim_proj, memory_key_padding_mask=x_mask.squeeze(1))
        pooled_output = decoder_output[0]
        decoder_out = self.projection_out(pooled_output) # torch.Size([16, 1024])
        return decoder_out

class TDNNPooling(nn.Module):
    def __init__(self, ssl_hidden_channels, tdnn_dim=[], 
                        tdnn_kernel=[], tdnn_dilation=[], conv_kernel=[], 
                        conv_stride=[], xvector_output_dim=512, pooling_hidden=512, **kwargs):
        super(TDNNPooling, self).__init__()

        self.tdnn_kernel = tdnn_kernel
        self.tdnn_dilation = tdnn_dilation
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride

        num_layers = 25
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(ssl_hidden_channels, tdnn_dim[0])

        tdnn_layers = [TDNNLayer(tdnn_dim, tdnn_kernel, tdnn_dilation, i) for i in range(len(tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # self.bos_idx = 0
        # embedding_pos = nn.Embedding(num_embeddings=1, embedding_dim=1500, padding_idx=0)
        # pos_encoder = SinusoidalPositionalEncoding(1500)
        # self.frontend_decoder = nn.Sequential(embedding_pos, pos_encoder)

        # decoder_layer = nn.TransformerDecoderLayer(d_model=1500, nhead=6)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        # self.projection_out = nn.Linear(1500, 1024)
        self.feature_extractor = nn.Linear(tdnn_dim[-1] * 2, pooling_hidden)

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=False):
        """
        Computes the output length of the convolutional layers
        """
        add_adapter = False
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.conv_kernel, self.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(3):
                input_lengths = _conv_out_length(input_lengths, 1, 2)
        return input_lengths

    def _get_pooling_tokens(self, batch_size, device):
        """Generate the initial pooling tokens (BOS tokens)."""
        return torch.full(
            (1, batch_size), self.bos_idx, dtype=torch.long, device=device
        )

    def _get_tdnn_output_lengths(self, input_lengths):
        """
        Computes the output length of the TDNN layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths    

    def _get_pooling_tokens(self, batch_size, device):
        """Generate the initial pooling tokens (BOS tokens)."""
        return torch.full((1, batch_size), self.bos_idx, dtype=torch.long, device=device)

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None): # x [b, 2028]
        norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
        x = (x * norm_weights.view(-1, 1, 1)).sum(dim=1)
        hidden_states = self.projector(x) # torch.Size([16, 313, 512])
        
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states) # torch.Size([16, 299, 1500])

        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1) # torch.Size([16, 1500])
            std_features = hidden_states.std(dim=1) # torch.Size([16, 1500])
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                length = int(length)
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1) # torch.Size([16, 3000])
        
        output_embeddings = self.feature_extractor(statistic_pooling) # torch.Size([16, 512])
        return output_embeddings

class Baseline(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(Baseline, self).__init__()
        
        self.eps = 1e-5
        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        # num_layers = 25
        # self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # self.projector = nn.Linear(ssl_hidden_channels, ssl_hidden_channels)
        
        self.x_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
        # norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
        # x = (x * norm_weights.view(-1, 1, 1)).sum(dim=1)
        # audio = self.projector(x) # torch.Size([16, 313, 512]) [batch_size, audio_len, hid_dim]

        decoder_out = self.x_pooling(x, x_lengths) # [batch_size, hid_dim]
        return decoder_out

class VATTTry1(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(VATTTry1, self).__init__()
        
        self.eps = 1e-5
        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        #self.emb_g = nn.Linear(spk_embed_dim, ssl_hidden_channels)
        # num_layers = 25
        # self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # self.projector = nn.Linear(ssl_hidden_channels, ssl_hidden_channels)
        
        #self.text_representations = TextRepresentations(39818, 1024, 4, 16, 2048, 0.2, 0, 5000)
        self.cross_attention = nn.ModuleList([CrossAttentionLayer(1024, 16, 2048, 0.2) for _ in range(6)])
        #self.self_attention_audio = nn.ModuleList([CrossAttentionLayer(1024, 16, 2048, 0.2) for _ in range(6)])

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.

        Arguments
        ---------
        shape_of_tensor : torch.Tensor
            It represents the size of tensor for generating Gaussian noise.
        device : str
            Device on which to perform computations.

        Returns
        -------
        gnoise : torch.Tensor
            The Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
        #spk_embeds = self.emb_g(F.normalize(spk_embeds)).unsqueeze(1) # torch.Size([11, 1, 1024])
        #audio = torch.cat([spk_embeds, x], dim=1) #self.projector(x) # torch.Size([16, 313, 512]) [batch_size, audio_len, hid_dim]
        #x_lengths = x_lengths + 1
        audio = x

        # for layer in self.self_attention_audio:
        #     audio = layer(audio, audio)

        text = txt_embeds #self.text_representations(txt_embeds) # [batch_size, src_len, hid_dim]

        for layer in self.cross_attention:
            text = layer(text, audio)
        decoder_out_txt = text[:, 0, :] # [batch_size, hid_dim]

        # mean_features = []
        # for i, length in enumerate(x_lengths):
        #     length = int(length)
        #     mean_features.append(audio[i, :length].mean(dim=0))
        # x_pooled = torch.stack(mean_features)
        # gnoise = self._get_gauss_noise(x_pooled.size(), device=x_pooled.device)
        # x_pooled += gnoise

        decoder_out = decoder_out_txt #torch.cat([x_pooled, decoder_out_txt], dim=-1)
        return decoder_out

class VATTTry2(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(VATTTry2, self).__init__()
        
        self.eps = 1e-5
        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        self.emb_g = nn.Linear(spk_embed_dim, ssl_hidden_channels)
        # num_layers = 25
        # self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # self.projector = nn.Linear(ssl_hidden_channels, ssl_hidden_channels)
        
        self.text_representations = TextRepresentations(39818, 1024, 4, 16, 2048, 0.2, 0, 5000)
        self.cross_attention = nn.ModuleList([CrossAttentionLayer(1024, 8, 2048, 0.2) for _ in range(2)])
        self.self_attention_audio = nn.ModuleList([CrossAttentionLayer(1024, 8, 2048, 0.2) for _ in range(2)])

        self.x_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
        spk_embeds = self.emb_g(F.normalize(spk_embeds)).unsqueeze(1) # torch.Size([11, 1, 1024])
        audio = torch.cat([spk_embeds, x], dim=1) #self.projector(x) # torch.Size([16, 313, 512]) [batch_size, audio_len, hid_dim]
        x_lengths = x_lengths + 1

        for layer in self.self_attention_audio:
            audio = layer(audio, audio)

        text = self.text_representations(txt_embeds) # [batch_size, src_len, hid_dim]
        for layer in self.cross_attention:
            text = layer(text, audio)
        decoder_out_txt = text[:, 0, :] # [batch_size, hid_dim]

        x_pooled = self.x_pooling(audio, x_lengths)

        decoder_out = torch.cat([x_pooled, decoder_out_txt], dim=-1)
        return decoder_out

class VATTTry3(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(VATTTry3, self).__init__()
        
        self.eps = 1e-5
        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        print("Using VATTTry3")

        # self.emb_g = nn.Linear(spk_embed_dim, ssl_hidden_channels)
        # num_layers = 25
        # self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # self.projector = nn.Linear(ssl_hidden_channels, ssl_hidden_channels)
        
        self.text_representations = TextRepresentations(39818, 1024, 4, 16, 2048, 0.2, 0, 5000)
        self.cross_attention_txt = nn.ModuleList([CrossAttentionLayer(1024, 16, 2048, 0.3) for _ in range(4)])
        self.cross_attention_audio = nn.ModuleList([CrossAttentionLayer(1024, 16, 2048, 0.3) for _ in range(4)])

        self.x_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)
        self.txt_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
        text = self.text_representations(txt_embeds) # [batch_size, src_len, hid_dim]
        audio = x
        audio_ = audio

        for layer in self.cross_attention_audio:
            audio = layer(audio, text)

        for layer in self.cross_attention_txt:
            text = layer(text, audio_)
        
        # #decoder_out_txt = text[:, 0, :] # [batch_size, hid_dim]
        txt_lengths = torch.tensor(commons.compute_length_from_mask(txt_masks)).cuda(non_blocking=True)

        x_pooled = self.x_pooling(audio, x_lengths)
        txt_pooled = self.txt_pooling(text, txt_lengths)

        decoder_out = torch.cat([x_pooled, txt_pooled], dim=-1)
        return decoder_out

class VATTTry4(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(VATTTry4, self).__init__()
        
        self.eps = 1e-5
        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        # self.emb_g = nn.Linear(spk_embed_dim, ssl_hidden_channels)
        self.dim_pred = DimensionPredictorVITS(ssl_hidden_channels, filter_channels, kernel_size, 
                                                    p_dropout, out_channel=d_dim, 
                                                    gin_channels=spk_embed_dim)

        
        self.text_representations = TextRepresentations(39818, 1024, 4, 16, 2048, 0.2, 0, 5000)
        self.cross_attention = nn.ModuleList([CrossAttentionLayer(1024, 8, 2048, 0.2) for _ in range(2)])
        self.cross_attention_audio = nn.ModuleList([CrossAttentionLayer(1024, 8, 2048, 0.2) for _ in range(2)])

        self.x_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)
        self.txt_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
        spk_embeds = spk_embeds.unsqueeze(-1)   # [B, spk_embed_dim, 1]
        x = x.transpose(1, 2)                   # [B, hidden_channels, T]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype) # [B, 1, T] 
        audio = self.dim_pred(x, x_mask, g=spk_embeds, txt=txt_embeds) # [B, pooling_hidden, T] 
        audio = audio.transpose(1, 2) # [B, T, hidden_channels]
        
        text = self.text_representations(txt_embeds) # [batch_size, src_len, hid_dim]
        audio_ = audio

        for layer in self.cross_attention_audio:
            audio = layer(audio, text)

        for layer in self.cross_attention:
            text = layer(text, audio_)
        
        #decoder_out_txt = text[:, 0, :] # [batch_size, hid_dim]
        txt_lengths = torch.tensor(commons.compute_length_from_mask(txt_masks)).cuda(non_blocking=True)

        x_pooled = self.x_pooling(audio, x_lengths)
        txt_pooled = self.txt_pooling(text, txt_lengths)

        decoder_out = torch.cat([x_pooled, txt_pooled], dim=-1)
        return decoder_out

class VATTTry5(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(VATTTry5, self).__init__()
        
        self.eps = 1e-5
        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        print("Using VATTTry5")

        self.emb_g = nn.Linear(spk_embed_dim, ssl_hidden_channels)
        # num_layers = 25
        # self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # self.projector = nn.Linear(ssl_hidden_channels, ssl_hidden_channels)
        
        #self.text_representations = TextRepresentations(39818, 1024, 4, 16, 2048, 0.2, 0, 5000)
        self.text_attention = nn.ModuleList([CrossAttentionLayer(1024, 8, 2048, 0.2) for _ in range(6)])

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.

        Arguments
        ---------
        shape_of_tensor : torch.Tensor
            It represents the size of tensor for generating Gaussian noise.
        device : str
            Device on which to perform computations.

        Returns
        -------
        gnoise : torch.Tensor
            The Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
        spk_embeds = self.emb_g(F.normalize(spk_embeds))
        #text = self.text_representations(txt_embeds) # [batch_size, src_len, hid_dim]
        text = txt_embeds

        for layer in self.text_attention:
            text = layer(text, text)

        mean_features = []
        for i, length in enumerate(x_lengths):
            length = int(length)
            mean_features.append(x[i, :length].mean(dim=0))
        x_pooled = torch.stack(mean_features)
        gnoise = self._get_gauss_noise(x_pooled.size(), device=x_pooled.device)
        x_pooled += gnoise
        
        txt_lengths = torch.tensor(commons.compute_length_from_mask(txt_masks)).cuda(non_blocking=True)
        mean_features = []
        for i, length in enumerate(txt_lengths):
            length = int(length)
            mean_features.append(text[i, :length].mean(dim=0))
        text_pooled = torch.stack(mean_features)
        gnoise = self._get_gauss_noise(text_pooled.size(), device=text_pooled.device)
        text_pooled += gnoise

        decoder_out = x_pooled + text_pooled + spk_embeds
        return decoder_out


class VATTTry6(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
        super(VATTTry6, self).__init__()

        print("Using VATTTry6")
        
        self.eps = 1e-5
        self.spk_embed_dim = spk_embed_dim  
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        d_model = d_dim
        combined_dim = d_model * 2
        
        self.audio_encoder = nn.Conv1d(ssl_hidden_channels, d_model, 3, padding=1, bias=False)
        self.text_encoder = nn.Conv1d(ssl_hidden_channels, d_model, 3, padding=1, bias=False)

        self.cross_audio_with_text = CrossmodalTransformer(n_layers=n_layers, n_heads=n_heads, d_model=d_model,
                                        attn_dropout=0.2, relu_dropout=0.1, emb_dropout=0.2,
                                        res_dropout=0.1, attn_mask=False, scale_embedding=True)
        self.cross_text_with_audio = CrossmodalTransformer(n_layers=n_layers, n_heads=n_heads, d_model=d_model,
                                        attn_dropout=0.2, relu_dropout=0.1, emb_dropout=0.2,
                                        res_dropout=0.1, attn_mask=False, scale_embedding=True)

        self.audio_layers = CrossmodalTransformer(n_layers=n_layers, n_heads=n_heads, d_model=d_model,
                                        attn_dropout=0.2, relu_dropout=0.1, emb_dropout=0.2,
                                        res_dropout=0.1, attn_mask=False, scale_embedding=True)
        self.text_layers = CrossmodalTransformer(n_layers=n_layers, n_heads=n_heads, d_model=d_model,
                                        attn_dropout=0.2, relu_dropout=0.1, emb_dropout=0.2,
                                        res_dropout=0.1, attn_mask=False, scale_embedding=True)

        self.fc1 = nn.Linear(combined_dim, combined_dim)
        self.fc2 = nn.Linear(combined_dim, combined_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
        x_mask = commons.sequence_mask(x_lengths, x.size(1)).to(x.dtype)
        txt_masks = txt_masks.to(x.dtype)

        x_audio = self.audio_encoder(x.transpose(1, 2)).transpose(1, 2)  # [batch_size, t, hid_dim]
        x_text = self.text_encoder(txt_embeds.transpose(1, 2)).transpose(1, 2) # [batch_size, t, hid_dim]

        # crossmodal attention
        x_audio = self.cross_audio_with_text(x_audio, x_text, txt_masks).transpose(0, 1) # (B, T, hid_dim)
        x_text = self.cross_text_with_audio(x_text, x_audio, x_mask).transpose(0, 1) # (B, T, hid_dim)

        # self-attention
        x_audio = self.audio_layers(x_audio, key_mask=x_mask)       # (t, B, hid_dim)
        x_text = self.text_layers(x_text, key_mask=txt_masks)       # (t, B, hid_dim)

        features = []
        for idx, (cur_a_mask, cur_t_mask) in enumerate(zip(x_mask, txt_masks)):
            cur_a_mask = cur_a_mask.long()
            cur_t_mask = cur_t_mask.long()

            cur_x_audio = x_audio[~cur_a_mask, idx, :].mean(dim=0).unsqueeze(0)
            cur_x_text = x_text[~cur_t_mask, idx, :].mean(dim=0).unsqueeze(0)
            features.append(torch.cat([cur_x_audio, cur_x_text], dim=1))
        features = torch.cat(features, dim=0)
        decoder_out = features + self.fc2(self.dropout(F.relu(self.fc1(features))))

        return decoder_out

# class ImageBindModel(nn.Module):
#     def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
#                        n_heads, n_layers, pooling_hidden,
#                        kernel_size, p_dropout, temporal_pred="GlowDP", spk_embed_dim=512, txt_embed_dim=1024, **kwargs):
#         super(ImageBindModel, self).__init__()
        
#         self.eps = 1e-5
#         self.spk_embed_dim = spk_embed_dim  
#         self.temporal_pred = temporal_pred
#         self.ssl_hidden_channels = ssl_hidden_channels

#         self.layer_norm_x = nn.LayerNorm(ssl_hidden_channels)
#         self.layer_norm_txt = nn.LayerNorm(ssl_hidden_channels)
        
#         self.x_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)
#         self.txt_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)

#         modality_trunks[ModalityType.TEXT] = instantiate_trunk(
#             text_embed_dim,
#             text_num_blocks,
#             text_num_heads,
#             pre_transformer_ln=False,
#             add_bias_kv=False,
#             drop_path=0.0,
#         )
#         self.modality_trunks_txt = SimpleTransformer(
#                                         embed_dim=embed_dim,
#                                         num_blocks=num_blocks,
#                                         ffn_dropout_rate=0.0,
#                                         drop_path_rate=drop_path,
#                                         attn_target=partial(
#                                             MultiheadAttention,
#                                             embed_dim=embed_dim,
#                                             num_heads=num_heads,
#                                             bias=True,
#                                             add_bias_kv=add_bias_kv,
#                                         ),
#                                         pre_transformer_layer=nn.Sequential(
#                                             nn.LayerNorm(embed_dim, eps=1e-6)
#                                             if pre_transformer_ln
#                                             else nn.Identity(),
#                                             EinOpsRearrange("b l d -> l b d"),
#                                         ),
#                                         post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
#                                     )
        
#         modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
#             audio_embed_dim,
#             audio_num_blocks,
#             audio_num_heads,
#             pre_transformer_ln=False,
#             add_bias_kv=True,
#             drop_path=audio_drop_path,
#         )
#         self.modality_trunks_audio = SimpleTransformer(
#                                         embed_dim=embed_dim,
#                                         num_blocks=num_blocks,
#                                         ffn_dropout_rate=0.0,
#                                         drop_path_rate=drop_path,
#                                         attn_target=partial(
#                                             MultiheadAttention,
#                                             embed_dim=embed_dim,
#                                             num_heads=num_heads,
#                                             bias=True,
#                                             add_bias_kv=add_bias_kv,
#                                         ),
#                                         pre_transformer_layer=nn.Sequential(
#                                             nn.LayerNorm(embed_dim, eps=1e-6)
#                                             if pre_transformer_ln
#                                             else nn.Identity(),
#                                             EinOpsRearrange("b l d -> l b d"),
#                                         ),
#                                         post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
#                                     )



#         modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
#             proj=nn.Sequential(
#                 nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
#                 nn.Linear(text_embed_dim, out_embed_dim, bias=False),
#             )
#         )

#         modality_heads[ModalityType.AUDIO] = nn.Sequential(
#             nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
#             SelectElement(index=0),
#             nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
#         )

#         modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
#             Normalize(dim=-1), LearnableLogitScaling(learnable=True)
#         )
#         modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
#             Normalize(dim=-1),
#             LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
#         )


#     def forward(self, x, x_lengths, attention_mask=None, spk_embeds=None, txt_embeds=None, txt_masks=None): # x [b, 2028] or [11, 25, 406, 1024])
#         hidden_states = x
#         hidden_states = self.layer_norm_x(hidden_states)
#         txt_embeds = self.layer_norm_txt(txt_embeds)
#         txt_masks = torch.sum(txt_masks, dim=1).long()

#         x_pooled = self.x_pooling(hidden_states, x_lengths)
#         txt_pooled = self.txt_pooling(txt_embeds, txt_masks)

#         decoder_out = torch.cat([x_pooled, txt_pooled], dim=1)
        #return decoder_out