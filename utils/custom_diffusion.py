# Modified from https://github.com/adobe-research/custom-diffusion/blob/main/src/diffuser_training.py
# which is under the Adobe Research License

# Copyright 2022, Adobe Inc. and its licensors. All rights reserved.

# ADOBE RESEARCH LICENSE

# Adobe grants any person or entity ("you" or "your") obtaining a copy of these certain research 
# materials that are owned by Adobe ("Licensed Materials") a nonexclusive, worldwide, royalty-free, 
# revocable, fully paid license to (A) reproduce, use, modify, and publicly display the Licensed 
# Materials; and (B) redistribute the Licensed Materials, and modifications or derivative works 
# thereof, provided the following conditions are met:

#     - The rights granted herein may be exercised for noncommercial research purposes (i.e., 
#       academic research and teaching) only. Noncommercial research purposes do not include 
#       commercial licensing or distribution, development of commercial products, or any other 
#       activity that results in commercial gain.
#     - You may add your own copyright statement to your modifications and/or provide additional 
#       or different license terms for use, reproduction, modification, public display, and 
#       redistribution of your modifications and derivative works, provided that such license terms 
#       limit the use, reproduction, modification, public display, and redistribution of such 
#       modifications and derivative works to noncommercial research purposes only.
#     - You acknowledge that Adobe and its licensors own all right, title, and interest in the 
#       Licensed Materials.
#     - All copies of the Licensed Materials must include the above copyright notice, this list of 
#       conditions, and the disclaimer below.
      
# Failure to meet any of the above conditions will automatically terminate the rights granted herein. 
 
# THE LICENSED MATERIALS ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE ENTIRE RISK AS TO THE USE,
# RESULTS, AND PERFORMANCE OF THE LICENSED MATERIALS IS ASSUMED BY YOU. ADOBE DISCLAIMS ALL WARRANTIES, 
# EXPRESS, IMPLIED OR STATUTORY, WITH REGARD TO YOUR USE OF THE LICENSED MATERIALS, INCLUDING, BUT NOT 
# LIMITED TO, NONINFRINGEMENT OF THIRD-PARTY RIGHTS. IN NO EVENT WILL ADOBE BE LIABLE FOR ANY ACTUAL, 
# INCIDENTAL, SPECIAL OR CONSEQUENTIAL DAMAGES, INCLUDING WITHOUT LIMITATION, LOSS OF PROFITS OR OTHER 
# COMMERCIAL LOSS, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE LICENSED MATERIALS, 
# EVEN IF ADOBE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

from diffusers.models.attention import CrossAttention

def unet_change_forward(unet):

    def new_forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        crossattn = False
        if context is not None:
            crossattn = True

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        if crossattn:
            modifier = torch.ones_like(key)
            modifier[:, :1, :] = modifier[:, :1, :]*0.
            key = modifier*key + (1-modifier)*key.detach()
            value = modifier*value + (1-modifier)*value.detach()

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def change_forward(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer)

    change_forward(unet)
    
    return unet
