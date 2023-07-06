import os
import shutil
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

in_volume = " NULL"
in_volume_bbox = " NULL "
in_genus= " 0"
tet_limit=" 100"
beta=" 0"
smooth_bef_loop=" 500"
smooth_n_loop=" 2"
smooth_in_loop=" 200"

data_dir = r"D:\projects\SparseRecon\IMG\6007\contour_points"
main_file = r"D:\projects\CycleGrouping_v1.0.0\exe_data\CycleGrouping_parallel.exe "
out_dir = os.path.join(data_dir, "cycle_group_recon") 
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

sample_steps = [5, 8, 10, 12, 15]
# sample_steps = [8]
for s_step in sample_steps:
    sample_dir = os.path.join(data_dir, "opt_cv_contour_plane_" + str(s_step))
    out_sub_dir = os.path.join(out_dir, "opt_cv_contour_plane_" + str(s_step))
    if not os.path.exists(out_sub_dir):
        os.mkdir(out_sub_dir)
    # sample_dir = r"D:\projects\CycleGrouping_v1.0.0\exe_data\Data\cv_contour_planes\contour_dist_8"
    files = os.listdir(sample_dir)
    final_file_dir = os.path.join(out_sub_dir, "recon_100")
    temp_file_dir = os.path.join(out_sub_dir, "temp")
    if not os.path.exists(final_file_dir):
        os.mkdir(final_file_dir)
    if not os.path.exists(temp_file_dir):
        os.mkdir(temp_file_dir)
    for file in files:
        if file.split('.')[-1] != "contour":
            continue
        id_str = file.split('.')[0]
        id_str = id_str.split('_')[-1]
        id = int(id_str)
        in_contour = os.path.join(sample_dir, file)
        print("process data : ", in_contour)
        sys_cmd = main_file + in_contour + in_volume + in_volume_bbox + temp_file_dir + tet_limit + beta + smooth_bef_loop + smooth_n_loop + smooth_in_loop + in_genus
        #%exe% %in_contour% %in_volume% %in_volume_bbox% %out_dir% %tet_limit% %beta% %smooth_bef_loop% %smooth_n_loop% %smooth_in_loop% %in_genus% 
        print(sys_cmd)
        try:
            os.system(sys_cmd)
        except:
            print("system command excuted failed!! ")
        out_files = os.listdir(temp_file_dir)
        
        for out_f in out_files:
            src = os.path.join(temp_file_dir, out_f)
            f_splits = out_f.split('.')
            new_f = f_splits[0] + "_" + id_str + '.' + f_splits[1]
            dst = os.path.join(final_file_dir, new_f)
            shutil.copy2(src, dst)
