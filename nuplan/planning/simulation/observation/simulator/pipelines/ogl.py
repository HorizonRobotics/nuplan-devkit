from nuplan.planning.simulation.observation.simulator.models.unet import UNet
from nuplan.planning.simulation.observation.simulator.models.texture import PointTexture
from nuplan.planning.simulation.observation.simulator.utils.train import load_model_checkpoint

def get_net():
    net = UNet(
        num_input_channels=8, 
        num_output_channels=3,
        feature_scale=4,
        num_res=4
        )
    return net


def get_texture(num_channels, size, texture_ckpt=None):
    texture = PointTexture(num_channels, size)
    if texture_ckpt is not None:
        texture = load_model_checkpoint(texture_ckpt, texture)    
    return texture


def backward_compat(args):
    if not hasattr(args, 'input_channels'):
        args.input_channels = None
    if not hasattr(args, 'conv_block'):
        args.conv_block = 'gated'

    if args.pipeline == 'READ.pipelines.ogl.Pix2PixPipeline':
        if not hasattr(args, 'input_modality'):
            args.input_modality = 1

    return args