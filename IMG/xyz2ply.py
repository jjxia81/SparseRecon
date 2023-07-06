import os

def get_points(file_name):
    points = []
    with open(file_name) as f:
        lines = f.readlines()
        # for line in lines:
        #     p_str = line.split(' ')
        return lines
    return None        

def write_ply(point_lines, out_file):
    with open(out_file, 'w') as f:
        ply_head = "ply\nformat ascii 1.0\ncomment VCGLIB generated\n"
        ply_head += "element vertex "+ str(len(point_lines)) + "\n"
        ply_head += "property float x\nproperty float y\nproperty float z\n"
        ply_head += "element face 0\n"
        ply_head += "property list uchar int vertex_indices\n"
        ply_head += "end_header\n"
        # print(ply_head)
        f.write(ply_head)

        for p in  point_lines:
            # line = str(p[0]) +" " + str(p[1]) + " " + str(p[2]) + "\n"
            f.write(p)
            # print(p)
            # break

if __name__ == "__main__" :
    file_dir = r"D:\projects\SparseRecon\IMG\6007\contour_points"
    sample_steps = [5, 8, 10, 12, 15]

    for s_step in sample_steps:
        sample_dir = os.path.join(file_dir, "opt_cv_contour_" + str(s_step))
        files = os.listdir(sample_dir)
        for file in files:
            if file.split('.')[1] != 'xyz':
                continue
            out_file = file.split('.')[0]
            out_file += ".ply"

            file_name = os.path.join(sample_dir, file)
            point_lines = get_points(file_name)
            out_file = os.path.join(sample_dir, out_file)
            write_ply(point_lines, out_file)
            print("save to file %s succeed", out_file)

