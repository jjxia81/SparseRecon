import numpy as np
from skimage import measure
import cv2 
import os

def write_ply(verts, faces, out_file):
    with open(out_file, 'w') as f:
        ply_head = "ply\nformat ascii 1.0\ncomment VCGLIB generated\n"
        ply_head += "element vertex "+ str(len(verts)) + "\n"
        ply_head += "property float x\nproperty float y\nproperty float z\n"
        ply_head += "element face "  +  str(len(faces)) + "\n"
        ply_head += "property list uchar int vertex_indices\n"
        ply_head += "end_header\n"
        # print(ply_head)
        f.write(ply_head)
        for point in verts:
            line =  str(point[2]) + " " + str(point[1]) + " " + str(point[0]-1) +"\n"
            # line =  str(point[0]) + " " + str(point[1]) + " " + str(point[2]) +"\n"
            f.write(line)
        for face in faces:
            line = "3 " + str(face[0]) +" " + str(face[1]) + " " + str(face[2]) + "\n"
            f.write(line)

def get_img_data(path):
    images = []
    ret, images = cv2.imreadmulti(path, images, cv2.IMREAD_ANYDEPTH)
    img_stacks = []
    if len(images) > 1:
        for img in images:
            # new_img = np.float32(mask) * 1.0
            new_img = img.astype(np.float32)
            img_stacks.append(new_img)
        padding_img = np.zeros_like(img_stacks[0])
        img_stacks.insert(0, padding_img)
        img_stacks.append(padding_img)
        
        mrc_data = np.stack(img_stacks)
        print(mrc_data.shape)
        return mrc_data
     
def reconstruct_mesh_with_marching_cubes(file_path, out_path):
    volume_data = get_img_data(file_path)
    verts, faces, normals, values = measure.marching_cubes(volume_data, 0.99)
    write_ply(verts, faces, out_path)

def reconstruction_all_files(input_dir, out_dir):
    file_names = os.listdir(input_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for f_name in file_names:
        if f_name.split('.')[-1] != 'tif':
            continue
        file_path = os.path.join(input_dir, f_name)
        out_name = f_name.split('.')[0] + '.ply'
        out_path = os.path.join(out_dir, out_name)
        reconstruct_mesh_with_marching_cubes(file_path, out_path)

def run_reconstruct():
    in_dir  = r'D:\projects\ET_tools\Files\data\img_stacks'
    out_dir = r'D:\projects\ET_tools\Files\data\marching_cubes_output'
    reconstruction_all_files(in_dir, out_dir)

if __name__ == "__main__":
    run_reconstruct()




