'''
Render multi-view image for object with texture

Support for rgb, depth, normal, mask
'''
from multiprocessing import Pool, Process
import argparse, sys, os, time
import logging
import numpy as np
import math
from math import radians

logging.basicConfig(filename='log.txt',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=12,
                    help='number of views to be rendered')
parser.add_argument('input_folder', type=str,
                    help='The path to where obj file and texture file stored')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--number_process', type=int, default=4,
                    help='number of multi-processing.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.0,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

import bpy
import pdb

# render main function
def render_function(model_dir, texture_dir):
    ### setting
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    # bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_environment = True
    bpy.context.scene.render.image_settings.file_format = args.format
    bpy.context.scene.render.image_settings.color_depth = args.color_depth

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        normalize = tree.nodes.new(type="CompositorNodeNormalize")
        links.new(render_layers.outputs['Depth'], normalize.inputs[0])
        links.new(normalize.outputs[0], depth_file_output.inputs[0])

    scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    scale_normal.blend_type = 'MULTIPLY'
    # scale_normal.use_alpha = True
    scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

    bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    bias_normal.blend_type = 'ADD'
    # bias_normal.use_alpha = True
    bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_normal.outputs[0], bias_normal.inputs[1])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    links.new(render_layers.outputs['Env'], albedo_file_output.inputs[0])

    # Delete default cube
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()
    bpy.data.objects['Lamp'].select = True
    bpy.ops.object.delete()
    
    ## render
    model_ids = os.listdir(model_dir)
    if '.DS_Store' in model_ids: model_ids.remove('.DS_Store')
    for index, model_id in enumerate(model_ids):
        model_id = model_id.split('.')[0]
        obj_file = os.path.join(model_dir, model_id+'.obj')
        texture_file = os.path.join(texture_dir, model_id+'.png')
        
        try: bpy.ops.import_scene.obj(filepath=obj_file)
        except: continue
        
        bpy.context.scene.render.engine = 'CYCLES'
        for object in bpy.context.scene.objects:
            if object.name in ['Camera']:
                object.select = False
            else:
                object.select = False
                object.cycles_visibility.shadow = False
        
        bpy.data.worlds['World'].use_nodes = True
        bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value[0:3] = (0.75, 0.75, 0.75)
        
        def parent_obj_to_camera(b_camera):
            origin = (0, 0, 0)
            b_empty = bpy.data.objects.new("Empty", None)
            b_empty.location = origin
            b_camera.parent = b_empty  # setup parenting

            scn = bpy.context.scene
            scn.objects.link(b_empty)
            scn.objects.active = b_empty
            return b_empty

        scene = bpy.context.scene
        bpy.context.scene.cycles.samples = 20
        scene.render.resolution_x = 256 # 384
        scene.render.resolution_y = 256
        scene.render.resolution_percentage = 100
        scene.render.alpha_mode = 'TRANSPARENT'
        cam = scene.objects['Camera']
        cam.location = (0, 3.2, 0.8) # modified
        cam.data.angle = 0.9799147248268127
        cam_constraint = cam.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        b_empty = parent_obj_to_camera(cam)
        cam_constraint.target = b_empty

        
        # set lighting
        bpy.ops.object.lamp_add(type='SUN')
        sun = scene.objects['Sun']
        sun.location = (0, 2.0, 2.0)
        # sun.shadow_method = 'NOSHADOW'
        sun_constraint = sun.constraints.new(type='TRACK_TO')
        sun_constraint.target = b_empty

        bpy.ops.object.lamp_add(type='SUN')
        sun = scene.objects['Sun.001']
        sun.location = (1.73, -1.0, 2.0)
        # .shadow_method = 'NOSHADOW'
        sun_constraint = sun.constraints.new(type='TRACK_TO')
        sun_constraint.target = b_empty

        bpy.ops.object.lamp_add(type='SUN')
        sun = scene.objects['Sun.002']
        sun.location = (-1.73, -1.0, 2.0)
        # sun.shadow_method = 'NOSHADOW'
        sun_constraint = sun.constraints.new(type='TRACK_TO')
        sun_constraint.target = b_empty
        
        '''
        world = bpy.data.worlds['World']
        world.light_settings.use_ambient_occlusion = False
        world.light_settings.ao_factor = 0.3
        '''
        bpy.data.materials.new('UVTexture')
        bpy.data.materials['UVTexture'].use_nodes = True
        texture_tree = bpy.data.materials['UVTexture'].node_tree
        texture_links = texture_tree.links
        texture_node = texture_tree.nodes.new("ShaderNodeTexImage")
        texture_node.image = bpy.data.images.load(texture_file)
        texture_links.new(texture_node.outputs[0], texture_tree.nodes['Diffuse BSDF'].inputs[0])
        bpy.data.scenes['Scene'].render.layers['RenderLayer'].material_override = bpy.data.materials['UVTexture']

        fp = args.output_folder
        scene.render.image_settings.file_format = 'PNG'  # set output format to .png

        stepsize = 360.0 / args.views   # 45 degrees
        rotation_mode = 'XYZ'

        for output_node in [depth_file_output, normal_file_output, albedo_file_output]:
            output_node.base_path = ''
        b_empty.rotation_euler[2] += radians(330)
        
        # render image by views
        pose_dict = {}
        for i in range(12):
            print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
            save_dir = os.path.join(args.output_folder, model_id)
            if os.path.exists(save_dir) == False: os.makedirs(save_dir)
            
            scene.render.filepath = os.path.join(save_dir, 'image_' + '{0:03d}'.format(int(i * stepsize)))                              # rgb
            depth_file_output.file_slots[0].path = os.path.join(save_dir, 'depth_' + '{0:03d}'.format(int(i * stepsize)) + '_')         # depth
            normal_file_output.file_slots[0].path = os.path.join(save_dir, 'normal_' + '{0:03d}'.format(int(i * stepsize)) + '_')       # normal
            albedo_file_output.file_slots[0].path = os.path.join(save_dir, 'mask_' + '{0:03d}'.format(int(i * stepsize)) + '_')         # mask

            bpy.ops.render.render(write_still=True)  # render still
            b_empty.rotation_euler[2] += radians(stepsize)
        
        # clear sys        
        for object in bpy.context.scene.objects:
            if object.name in ['Camera']:
                object.select = False
            else:
                object.select = True
        bpy.ops.object.delete()  

###### render model images
model_dir = '/home/shunming/data/workshop/scripts/data_preparetion/final_dataset/reconstruction/train/normalized_model/'
texture_dir = '/home/shunming/data/workshop/scripts/data_preparetion/final_dataset/reconstruction/train/texture/'
render_function(model_dir, texture_dir)
