"""
normalize obj and remove same instance
"""
import os
import json
from tqdm import tqdm
import numpy as np
import copy
import shutil
from collections import Counter
import pdb
from tqdm import tqdm

def generate_veretx_group(file):
    with open(file, 'r') as f:
        vertex_group = []
        part_vertex = []
        last_fisrt = ''

        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            split_line = line.split(' ')
            curr_first = split_line[0]
            if curr_first != last_fisrt:
                if part_vertex != []: vertex_group += part_vertex
                part_vertex = []
            if 'v' == curr_first:
                try:
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                except:
                    continue
                    pdb.set_trace()
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                # part_vertex += vertex
                part_vertex.append(vertex)

            last_fisrt = curr_first

        # remove shadow vertex
        if len(vertex_group[-1]) == 4 and len(vertex_group) != 0:
            vertex_group.pop()
    return vertex_group


def normalize_vertex_group(vertex_group):
    all_vertex = vertex_group
    # find max axis
    min_val, max_val, all_vertex_numpy = find_max_axis(all_vertex)

    # norm
    # all_vertex_numpy = np.array(all_vertex)
    norm_all_vertex_numpy = 2 * (all_vertex_numpy - min_val) / (max_val - min_val) + (-1)
    #norm_all_vertex_numpy = norm_all_vertex_numpy*0.5
    #pdb.set_trace()
    return norm_all_vertex_numpy
    # # set back to vertex_group
    # norm_vertex_group = []
    # norm_all_vertex = norm_all_vertex_numpy.tolist()
    # for i, part_vertex in enumerate(vertex_group):
    #     if i == 0:
    #         cur_id = len(part_vertex)
    #         norm_part_vertex = norm_all_vertex[:cur_id]
    #     else:
    #         cur_id = len(part_vertex) + last_id
    #         norm_part_vertex = norm_all_vertex[last_id: cur_id]
    #     norm_vertex_group.append(norm_part_vertex)
    #     last_id = cur_id
    #
    # return norm_vertex_group


def find_max_axis(all_vertex):
    # find max axis
    all_vertex_np = np.array(all_vertex)
    def find_max_min(axis_vertex):
        min_num = min(axis_vertex)
        max_num = max(axis_vertex)
        return min_num, max_num

    x_min, x_max = find_max_min(all_vertex_np[:, 0])
    y_min, y_max = find_max_min(all_vertex_np[:, 1])
    z_min, z_max = find_max_min(all_vertex_np[:, 2])

    x_mean = (x_max + x_min) / 2
    y_mean = (y_max + y_min) / 2
    z_mean = (z_max + z_min) / 2
    all_vertex_np[:, 0] = all_vertex_np[:, 0] - x_mean
    all_vertex_np[:, 1] = all_vertex_np[:, 1] - y_mean
    all_vertex_np[:, 2] = all_vertex_np[:, 2] - z_mean

    x_min, x_max = find_max_min(all_vertex_np[:, 0])
    y_min, y_max = find_max_min(all_vertex_np[:, 1])
    z_min, z_max = find_max_min(all_vertex_np[:, 2])

    # def translate_nonzero(axis_min, axix_vertex):
    #     if axis_min < 0:
    #         axix_vertex += abs(axis_min)
    #         return axix_vertex
    # mimux = min(x_min, y_min, z_min)
    # all_vertex_np = translate_nonzero(mimux, all_vertex_np)
    #
    x_len = abs(x_min - x_max)
    y_len = abs(y_min - y_max)
    z_len = abs(z_min - z_max)
    max_num = max(x_len, y_len, z_len)

    if max_num == x_len:
        return x_min, x_max, all_vertex_np
    elif max_num == y_len:
        return y_min, y_max, all_vertex_np
    elif max_num == z_len:
        return z_min, z_max, all_vertex_np
    else:
        print('max_num can not be recongnised')
        exit(-1)


def replace_original_vertex(norm_vertex_group, file, save_file):
    all_norm_vertex = []
    for i, norm_part_vertex in enumerate(norm_vertex_group):
        all_norm_vertex.append(norm_part_vertex)

    norm_lines = []
    with open(file, 'r') as f:

        v_id = 0
        lines = f.readlines()
        try:
            for i, line in enumerate(lines):
                split_line = line.split(' ')
                curr_first = split_line[0]

                # check if shadow exist in line
                for element in split_line:
                    if 'shadow' in element:
                        break
                if 'shadow' in split_line[-1] or 'Shadow' in split_line[-1] or 'SHADOW' in split_line[-1]:
                    break

                if 'v' == curr_first:
                    norm_vertex = all_norm_vertex[v_id]
                    norm_line = 'v  ' + str(norm_vertex[0]) + ' ' + str(norm_vertex[1]) + ' ' + str(norm_vertex[2]) + '\n'
                    v_id += 1
                else:
                    norm_line = line
                norm_lines.append(norm_line)
            f.close()
        except:
            print(file)

    #pdb.set_trace()
    if len(norm_lines) < 200:
        return None
    split_file = file.split('/')

    with open(save_file, 'w') as f:
        for norm_line in norm_lines:
            f.write(norm_line)
        f.close()

def find_remove_same_obj(summary_list):
    for i in tqdm(range(len(summary_list))):
        for j in range(i, len(summary_list)):
            queue_vertex_group, queue_file = summary_list[j]
        # for j, (queue_vertex_group, queue_file) in enumerate(summary_list):
            if i == j: continue
            if i == len(summary_list):
                return summary_list
            refer_vertex_group = summary_list[i][0]

            # vertex_group number not the same continue
            if len(refer_vertex_group) != len(queue_vertex_group): continue

            # vertex_content not the same continue
            refer_all_vertex = []
            queue_all_vertex = []
            remove_list = []
            for refer_part_vertex in refer_vertex_group:
                refer_all_vertex += refer_part_vertex
            for queue_part_vertex in queue_vertex_group:
                queue_all_vertex += queue_part_vertex

            # if vertex group not the same, must be not the same obj
            if len(refer_all_vertex) != len(queue_all_vertex):continue

            # sort obj vertex and find same instance
            refer_all_vertex_clone = copy.deepcopy(refer_all_vertex)
            queue_all_vertex_clone = copy.deepcopy(queue_all_vertex)
            refer_all_vertex_clone.sort()
            queue_all_vertex_clone.sort()
            compare = np.array(refer_all_vertex_clone) == np.array(queue_all_vertex_clone)
            compare_list = compare.tolist()
            true_num = compare_list.count([True, True, True])
            total = int(compare.size / 3)
            percent = true_num / total

            if percent > 0.01:
                remove_list.append(summary_list[j])
            # if (np.array(refer_all_vertex) == np.array(queue_all_vertex)).all():
            #     remove_list.append([queue_vertex_group, queue_file])

        # # delete same obj
        if len(remove_list) == 0: continue
        for element in remove_list:
            print(element[1], summary_list[i][1])
            summary_list.remove(element)

            if element in summary_list:
                print('delet failed')

    # return summary_list

def check_create_folder(folder):
    if os.path.exists(folder) == False:
        os.mkdir(folder)

def copy_texture_file(refer_model_dir, target_texture_dir, save_dir):
    model_file_list = os.listdir(refer_model_dir)
    texture_file_list = os.listdir(target_texture_dir)

    for i, model_file in enumerate(model_file_list):
        split_model_file = model_file.split('.')
        model_id = split_model_file[0]
        if model_id + '.png' in texture_file_list:
            src_file = os.path.join(target_texture_dir, model_id + '.png')
            dst_file = os.path.join(save_dir, model_id + '.png')
            shutil.copy(src_file, dst_file)

def compute_norm_original_ratio(pix3d_original_vertex, pix3d_norml_vertex):
    def convert_list2numpy(input_list):
        output_array = []
        for ele in input_list:
            output_array.append(ele[0])
        output_array = np.array(output_array)
        return output_array

    pix3d_original_vertex = convert_list2numpy(pix3d_original_vertex)
    pix3d_norml_vertex = convert_list2numpy(pix3d_norml_vertex)

    print('pix3d_original_vertex')
    a = pix3d_original_vertex[:, 0]
    print(np.min(pix3d_original_vertex[:, 0]), np.max(pix3d_original_vertex[:, 0]))
    print(np.min(pix3d_original_vertex[:, 1]), np.max(pix3d_original_vertex[:, 1]))
    print(np.min(pix3d_original_vertex[:, 2]), np.max(pix3d_original_vertex[:, 2]))

    print('pix3d_norml_vertex')
    print(np.min(pix3d_norml_vertex[:, 0]), np.max(pix3d_norml_vertex[:, 0]))
    print(np.min(pix3d_norml_vertex[:, 1]), np.max(pix3d_norml_vertex[:, 1]))
    print(np.min(pix3d_norml_vertex[:, 2]), np.max(pix3d_norml_vertex[:, 2]))

def main():
    workshop_dir = '/home/public/IJCAI_2020_retrieval/validation/model/'
    save_dir = '/home/public/IJCAI_2020_retrieval/validation/normalized_model/'

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    shape_ids = os.listdir(workshop_dir)
    for shape_id in tqdm(shape_ids):
        model_file = os.path.join(workshop_dir, shape_id)
        save_file = os.path.join(save_dir, shape_id)
        try:
            # get obj vertex
            vertex_group = generate_veretx_group(model_file)
            norm_vertex_group = normalize_vertex_group(vertex_group)
        except:
            print(model_file)
            continue
        #pdb.set_trace()
        replace_original_vertex(norm_vertex_group.tolist(), model_file, save_file)


if __name__ == '__main__':
    main()

