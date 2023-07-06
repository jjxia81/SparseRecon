import os

# normal_exe_file = r'.\jet\normals_estimation.exe '
# data_dir = r"D:\projects\SparseRecon\IMG\6007\contour_points"
# outnormal_dir = "in" 
# xyz_files = os.listdir(data_dir)
# for file in xyz_files:
#     xyz_path = os.path.join(data_dir, file)
#     file_name = file.split('.')[0] + "_n.xyz"
#     xyz_out_path = os.path.join(outnormal_dir, file_name)

#     sys_cmd  = normal_exe_file + xyz_path + " " + xyz_out_path + " 6"
#     print(sys_cmd)
#     os.system(sys_cmd)
# coarse_dir = r"D:\projects\IsoConstraints\in\sample"

data_root_dir = r"D:\projects\SparseRecon\IMG\6007\contour_points"
fine_dir = r"D:\projects\IsoConstraints\in\dense"
main_file = r"D:\projects\ipsr-master\ipsr-master\Release\ipsr.exe "

out_dir = os.path.join(data_root_dir, "ipsr_recon")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
sample_steps = [5, 8, 10, 12, 15] 
for s_step in sample_steps:
    coarse_dir = os.path.join(data_root_dir, "opt_cv_contour_" + str(s_step))
    out_sub_dir = os.path.join(out_dir, "opt_cv_contour_" + str(s_step))
    if not os.path.exists(out_sub_dir):
        os.mkdir(out_sub_dir)
    coarse_files = os.listdir(coarse_dir)
    for i in range(len(coarse_files)):
        coarse_file = coarse_files[i]
        
        if coarse_file.split('.')[1] != "ply":
            continue
        file_out = coarse_file.split('.')[0] + "_ipsr.ply" 
        file_out = os.path.join(out_sub_dir, file_out)
        coarse_file = os.path.join(coarse_dir, coarse_file)
    
        print(coarse_file)
        sys_cmd = main_file + " --in " + coarse_file + " --out " + file_out 
        print(sys_cmd)
        os.system(sys_cmd)

'./jets/normals_estimation.exe input_xyz output_xyz neighbour_size '