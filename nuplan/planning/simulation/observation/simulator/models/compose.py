import torch
import torch.nn as nn


class NetAndTexture(nn.Module):
    def __init__(self, net, textures, supersampling=1, temporal_average=False):
        super().__init__()
        self.net = net
        self.ss = supersampling

        try:
            textures = dict(textures)
        except TypeError:
            textures = {0: textures}

        self._textures = {k: v.cpu() for k, v in textures.items()}
        self._loaded_textures = []

        self.last_input = None
        self.temporal_average = temporal_average

    def load_textures(self, texture_ids):
        if torch.is_tensor(texture_ids):
            texture_ids = texture_ids.cpu().tolist()
        elif isinstance(texture_ids, int):
            texture_ids = [texture_ids]

        for tid in texture_ids:
            self._modules[str(tid)] = self._textures[tid]
        self._loaded_textures = texture_ids

    def unload_textures(self):
        for tid in self._loaded_textures:
            self._modules[str(tid)].cpu()
            del self._modules[str(tid)]

    def reg_loss(self):
        loss = 0
        for tid in self._loaded_textures:
            loss += self._modules[str(tid)].reg_loss()

        return loss

    # def forward(self, inputs, **kwargs):
    #     outs = {'im_out':[]}
    #     # outs = {'x1':[],'x2':[],'x4':[],}
    #     texture_ids = inputs['id']
    #     del inputs['id']
    #     if torch.is_tensor(texture_ids):
    #         texture_ids = texture_ids.tolist()
    #     elif isinstance(texture_ids, int):
    #         texture_ids = [texture_ids]

    #     for i, tid in enumerate(texture_ids): # per item in batch
    #         input = {k: v[i][None] for k, v in inputs.items()}
    #         assert 'uv' in list(input)[0], 'first input must be uv'

    #         texture = self._modules[str(tid)]
    #         j = 0
    #         keys = list(input)
    #         input_multiscale = []
    #         while j < len(keys): # sample texture at multiple scales
    #             tex_sample = None
    #             input_ex = []
    #             if 'uv' in keys[j]:
    #                 tex_sample = texture(input[keys[j]])
    #                 j += 1
    #                 while j < len(keys) and 'uv' not in keys[j]:
    #                     input_ex.append(input[keys[j]])
    #                     j += 1
    #             assert tex_sample is not None
    #             input_cat = torch.cat(input_ex + [tex_sample], 1)


    #             if self.ss > 1:
    #                 input_cat = nn.functional.interpolate(input_cat, scale_factor=1./self.ss, mode='bilinear')

    #             input_multiscale.append(input_cat)
            
    #         if self.temporal_average:
    #             if self.last_input is not None:
    #                 for i in range(len(input_multiscale)):
    #                     input_multiscale[i] = (input_multiscale[i] + self.last_input[i]) / 2
    #             self.last_input = list(input_multiscale)

    #         out = self.net(*input_multiscale, **kwargs)
    #         outs['im_out'].append(out['im_out'])
    #         if 'seg_out' in out:
    #             if 'seg_out' not in outs:
    #                 outs['seg_out'] = []
    #             outs['seg_out'].append(out['seg_out'])


    #     if 'seg_out' in outs and len(outs['seg_out']) == len(outs['im_out']):
    #         outs['seg_out'] = torch.cat(outs['seg_out'], 0)
    #     outs['im_out'] = torch.cat(outs['im_out'], 0)


        
    #     if kwargs.get('return_input'):
    #         return outs, input_multiscale
    #     else:
    #         return outs
        

    def forward(self, inputs, **kwargs):
        outs = {'im_out':[]}
        # outs = {'x1':[],'x2':[],'x4':[],}
        texture_ids = inputs['id']
        del inputs['id']
        if torch.is_tensor(texture_ids):
            texture_ids = texture_ids.tolist()
        elif isinstance(texture_ids, int):
            texture_ids = [texture_ids]

        all_input_multiscale = []
        texture = self._modules['0']
        for i, tid in enumerate(texture_ids): # per item in batch
            input = {k: v[i][None] for k, v in inputs.items()}
            assert 'uv' in list(input)[0], 'first input must be uv'
            
            j = 0
            keys = list(input)
            input_multiscale = []
            while j < len(keys): # sample texture at multiple scales
                tex_sample = texture(input[keys[j]])
                j += 1
                if self.ss > 1:
                    input_cat = nn.functional.interpolate(input_cat, scale_factor=1./self.ss, mode='bilinear')
                input_multiscale.append(tex_sample)
            
            if self.temporal_average:
                if self.last_input is not None:
                    for i in range(len(input_multiscale)):
                        input_multiscale[i] = (input_multiscale[i] + self.last_input[i]) / 2
                self.last_input = list(input_multiscale)
            
            all_input_multiscale.append(input_multiscale)

        batch_input_multiscale = [torch.cat(items, dim=0) for items in zip(*all_input_multiscale)]
        out = self.net(*batch_input_multiscale, **kwargs)
        outs['im_out'] = out['im_out']
        
        if kwargs.get('return_input'):
            return outs, input_multiscale
        else:
            return outs