from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
#import numpy as np
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding
from fairseq.models import BaseFairseqModel

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor

from fairseq.models.nat import FairseqNATDecoder, FairseqNATEncoder, FairseqNATModel, ensemble_decoder, ensemble_encoder

class ErnieMEmbeddings(nn.Module):
    r"""
    Include embeddings from word, position.
    """
    
    def __init__(self,args):
        super(ErnieMEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        #past_key_values_length: int = 0,
    ):

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids is None:
            #input_shape = paddle.shape(inputs_embeds)[:-1]
            input_shape = inputs_embeds.shape[:-1]
            # maybe need use shape op to unify static graph and dynamic graph
            ones = torch.ones(list(input_shape), dtype=torch.int64, device = torch.device('cuda'))
            seq_length = torch.cumsum(ones, dim=1)
            #seq_length = torch.cumsum(ones, dim=0)
            position_ids = seq_length - ones

            '''
            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length
            '''
            position_ids.stop_gradient = True

        position_ids += 2

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErnieMPooler(nn.Module):
    def __init__(self,args):
        super(ErnieMPooler, self).__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class VanillaEncoder(FairseqNATEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        '''
        if hasattr(self.args, 'CAMLMload') and self.args.CAMLMload != "NULL" and self.args.CAMLMload != "First":
            print("*************************Using CAMLM pretrained encoder****************")
                
            pretrained_encoder, _ = checkpoint_utils.load_model_ensemble(utils.split_paths(self.args.CAMLMload))
            self.CAMLM = pretrained_encoder[0]
        '''
    @ensemble_encoder
    def forward(
            self,
            src_tokens,
            attn_mask: Optional[torch.Tensor] = None,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        if isinstance(attn_mask, torch.Tensor):
            attn_mask.stop_gradient = True
            attn_mask.detach()
        
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        encoder_pos = self.embed_positions(src_tokens)
        # account for padding while computing the representation
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)
        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None, attn_mask = attn_mask
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_pos": [encoder_pos],
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,
            "src_lengths": [],
        }

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_pos"]) == 0:
            new_encoder_pos = []
        else:
            new_encoder_pos = [
                encoder_out["encoder_pos"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_pos": new_encoder_pos,
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


@register_model('CAMLM')
class ErnieMModel(FairseqNATModel):#(BaseFairseqModel):#ErnieMPretrainedModel):
    '''
    def init_weights(self, layer, args):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            if isinstance(layer.weight, torch.Tensor):
                
                w = torch.normal(
                        mean=0.0,
                        std=0.02,
                        size=layer.weight.shape,
                    )
                layer.weight = nn.parameter(w)
    '''
    '''
    def __init__(self, args):
        super(ErnieMModel, self).__init__()
        self.pad_token_id = args.pad_token_id
        self.initializer_range = args.initializer_range
        self.embeddings = ErnieMEmbeddings(args)
        encoder_layer = nn.TransformerEncoderLayer(
            args.hidden_size,
            args.num_attention_heads,
            dim_feedforward=4 * args.hidden_size,
            dropout=args.hidden_dropout_prob,
            activation=args.hidden_act,
            batch_first=True,
            #attn_dropout=args.attention_probs_dropout_prob,
            #act_dropout=0,
            #normalize_before=False,
        )
        self.nums_heads = args.num_attention_heads
        self.encoder = nn.TransformerEncoder(encoder_layer, args.num_hidden_layers)
        self.pooler = ErnieMPooler(args)
        
        self.output = nn.Linear(args.hidden_size, args.vocab_size)
        #self.apply(self.init_weights)
'''
    @classmethod
    #def build_model(cls, args, tgt_dict, embed_tokens):
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        
        
        encoder = VanillaEncoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder
        #return ErnieMModel(args)
    
    def build_decoder(args, tgt_dict, embed_tokens):
        
        #decoder = VanillaDecoder(args, tgt_dict, embed_tokens)
        decoder = nn.Linear(args.decoder_embed_dim, tgt_dict.__len__())#args.vocab_size)
        
        return decoder
    
    def forward(
            self,
            src_tokens,
            attn_mask: Optional[torch.Tensor] = None,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,):
        
        ori_attention_mask = attn_mask.clone()
        nums_heads = 8
        
        multi_attention_mask = attn_mask.tile((nums_heads,1,1))
        
        multi_attention_mask.detach()
        
        encoder_outputs = self.encoder(src_tokens, multi_attention_mask)
        
        encoder_outputs = encoder_outputs["encoder_out"][0].transpose(0, 1)
        
        encoder_outputs = self.decoder(encoder_outputs)
        
        return (encoder_outputs, ori_attention_mask)
    '''
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        mask_loc: Optional[Tensor] = None,
        #past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        #use_cache: Optional[bool] = None,
        #output_hidden_states: Optional[bool] = None,
        #output_attentions: Optional[bool] = None,
        #return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `[batch_size, num_tokens]` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
            past_key_values (tuple(tuple(Tensor)), optional):
                The length of tuple equals to the number of layers, and each inner
                tuple haves 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`)
                which contains precomputed key and value hidden states of the attention blocks.
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`.
            tuple: Returns tuple (``sequence_output``, ``pooled_output``).

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieMModel, ErnieMTokenizer

                tokenizer = ErnieMTokenizer.from_pretrained('ernie-m-base')
                model = ErnieMModel.from_pretrained('ernie-m-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output = model(**inputs)

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        
        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False
        use_cache = use_cache if use_cache is not None else False

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        
        if attention_mask is None:
            # TODO(linjieccc): fix attention mask after uie-m related models updated
            if torch.is_floating_point(self.pooler.dense.weight):
                
                attention_mask = torch.unsqueeze(
                    #(input_ids == 0).astype(self.pooler.dense.weight.dtype) * -1e4, dim=[1, 2]
                    (input_ids == 0).float() * -1e4, dim=[1, 2]
                )
                
                attention_mask = torch.unsqueeze(torch.unsqueeze(
                    #(input_ids == 0).astype(self.pooler.dense.weight.dtype) * -1e4, dim=[1, 2]
                    (input_ids == 0).float() * -1e4, dim=1), dim = 1
                )
        
            
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = torch.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype, device = torch.device('cuda'))
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
            
        # For 2D attention_mask from tokenizer
        
       # elif attention_mask.ndim == 2:
            #if torch.get_default_dtype() == torch.float32():
            #default type is float 32
       #     attention_mask = torch.unsqueeze(torch.unsqueeze(attention_mask, dim=1), dim = 1).float()
            #attention_mask = torch.unsqueeze(attention_mask, dim=[1, 2]).astype(torch.get_default_dtype())
        #    attention_mask = (1.0 - attention_mask) * -1e4
        
        
        attention_mask.stop_gradient = True

        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            #past_key_values_length=past_key_values_length,
        )

        #embedding_output = embedding_output.transpose(0,1)
        
        #ori_attention_mask = attention_mask.copy()
        
        multi_attention_mask = attention_mask.tile((self.nums_heads,1,1))
        
        multi_attention_mask.detach()
        
       # self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            multi_attention_mask,
            #cache=past_key_values,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
        )
        #[L, Bsz, Hidden] -> [Bsz, L, Hidden]
        #encoder_outputs = encoder_outputs.transpose(0,1)
        
        #TODO use mask_loc to generate the final output
        
        encoder_outputs = self.output(encoder_outputs)
        
        
        #res = torch.zeros([encoder_outputs.shape[0],mask_loc.shape[1],encoder_outputs.shape[2]], device = torch.device('cuda'))
        
       # for i in range(mask_loc.shape[0]):
        #    for j in range(mask_loc.shape[1]):
         #       res[i][j] = encoder_outputs[i][mask_loc[i][j]]
                    
        
        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs
            
            #pooled_output = self.pooler(sequence_output)
            
            #select masked tokens
            return (encoder_outputs, attention_mask)
            
            #return (sequence_output, pooled_output)

        exit()
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        
        #pooled_output should be the final value
#        if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

'''

        
        

@register_model_architecture(
    "CAMLM", "CAMLM"
)
def base_architecture(args):
    #args.vocab_size = getattr(args, "vocab_size", 9712)#28848)#
    
    #args.hidden_size = getattr(args, "hidden_size", 512)
    #args.num_hidden_layers = getattr(args, "num_hidden_layers", 6)
    #args.num_attention_heads = getattr(args, "num_attention_heads", 8)
    #args.intermediate_size = getattr(args, "intermediate_size", 3072)
    '''
    args.hidden_act = getattr(args, "hidden_act", "gelu")
    args.hidden_dropout_prob = getattr(args, "hidden_dropout_prob", 0.1)
    args.attention_probs_dropout_prob = getattr(args, "attention_probs_dropout_prob", 0.1)
    args.max_position_embeddings = getattr(args, "max_position_embeddings", 770)
    args.type_vocab_size = getattr(args, "type_vocab_size", 16)
    args.pad_token_id = getattr(args, "pad_token_id", 1)
    args.initializer_range = getattr(args, "initializer_range", 0.02)
    ''' 
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)


