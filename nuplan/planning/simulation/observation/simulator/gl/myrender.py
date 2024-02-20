import torch
import pcpr
import numpy as np

inv = np.linalg.inv


class MyRender:
    def __init__(self, ds_list=None):
        if ds_list:
            self.update_ds(ds_list)
        
    def update_ds(self, ds_list):
        self.ds_list = ds_list
        self.ds_ids = [d.id for d in ds_list]
        self.tgt_sh = self.ds_list[0].tgt_sh
        self.points = {ds.id: torch.from_numpy(np.asarray(ds.scene_data['pointcloud']['xyz']).astype(np.float32)) for ds in ds_list}
        
    def render(self, data):
        input_format = self.ds_list[0].input_format.replace(' ', '').split(',')
        out_dict, depth_dict = {}, {}
        out_dict['id'] = data['input']['id']
        
        proj_matrix = data['proj_matrix'].numpy()
        view_matrix = data['view_matrix'].numpy()
        total_m = proj_matrix @ inv(view_matrix)
        total_m = total_m.astype(np.float32)
        total_m = torch.from_numpy(total_m)
       
        for i,k in enumerate(input_format):
            w = int(self.tgt_sh[0]*(0.5**i))
            h = int(self.tgt_sh[1]*(0.5**i))
            indexs, depths = torch.zeros(len(data['input']['id']),h,w),torch.zeros(len(data['input']['id']),h,w)
            for ds_id in self.ds_ids:
                idx = torch.where(data['input']['id']==ds_id)[0]
                index_buffer, depth_buffer =  pcpr.forward(self.points[ds_id], total_m[idx], w, h, 512)
                indexs[idx] = index_buffer
                depths[idx] = depth_buffer
            out_dict[k] = indexs.unsqueeze(1)
            depth_dict[k] = depths.unsqueeze(1)   
        return out_dict, depth_dict
    

class MyRenderSimple:
    def __init__(self, tgt_sh, input_format):
        self.tgt_sh = tgt_sh
        self.input_format = input_format.replace(' ', '').split(',')
 
    def render(self, data):
        out_dict, depth_dict = {}, {}
        out_dict['id'] = data['input']['id']
        
        proj_matrix = data['proj_matrix']
        view_matrix = data['view_matrix']
        total_m = proj_matrix @ inv(view_matrix)
        total_m = total_m.astype(np.float32)
        total_m = torch.from_numpy(total_m)
        # total_m = torch.from_numpy(total_m).unsqueeze(0)
        # total_m = torch.cat([total_m, total_m, total_m, total_m, total_m, total_m])
        points = torch.from_numpy(data['points'].astype(np.float32))
       
        for i,k in enumerate(self.input_format):
            w = int(self.tgt_sh[0]*(0.5**i))
            h = int(self.tgt_sh[1]*(0.5**i))
            indexs, depths = pcpr.forward(points, total_m, w, h, 512)
            out_dict[k] = indexs.unsqueeze(1)
            depth_dict[k] = depths.unsqueeze(1)   
        return out_dict, depth_dict
    
