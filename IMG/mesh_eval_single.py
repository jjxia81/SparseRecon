import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# data_dir = r"D:\data\mesh_evaluation\MeshVolume\6007_mesh_contour_overlap"

data_dir_1 = r"D:\data\output\6007_mesh_contour_overlap\mesh_evaluation_d5"

data_dir_2 = r"D:\data\output\6007_mesh_contour_overlap\mesh_evaluation_d7"

file_dict = ["marching_cubes"]
for i in range(20):
    file_name = "slice_" + str(i)
    file_dict.append(file_name)

# IOU_rates_vals_1 = []             
# IOU_rates_vals_2 = []
# for file in file_dict:
#     csv_file1 = os.path.join(data_dir_1, file + "\IOU_rate.csv")
#     df1 = pd.read_csv(csv_file1, header=None, sep=' ')  
#     val_1 = df1.mean()
#     IOU_rates_vals_1.append(val_1[1])

def get_IOU_rate(IOU_data_root):
    valid_labels = []
    iou_rates = []
    slice_labels = os.listdir(IOU_data_root)
    print("slice_labels : ", slice_labels)
    ordered_labels = []
    for i in range(len(slice_labels)):
        s_label = "slice_" + str(i)
        ordered_labels.append(s_label)

    invalid_labels = []
    for slice_file in ordered_labels:
        csv_file1 = os.path.join(IOU_data_root, slice_file + "\IOU_rate.csv")
        if not os.path.exists(csv_file1) :
            invalid_labels.append(slice_file)
            continue
        print(csv_file1)
        if os.path.getsize(csv_file1) == 0:
            invalid_labels.append(slice_file)
            continue
        df1 = pd.read_csv(csv_file1, header=None, sep=' ') 
        
        if df1.empty:
            invalid_labels.append(slice_file)
            continue 
        print("read csv succeed! ")
        val_1 = df1.mean()
        df1[~(df1 == 0).all(axis=1)]
        val_1 = df1.mean()
        iou_rates.append(val_1[1])
        valid_labels.append(slice_file)
    print("invalid files **************************** : ", invalid_labels)
    return valid_labels, iou_rates

def get_IOU_rate_dict(IOU_data_root):
    valid_labels = []
    iou_rates = []
    slice_labels = os.listdir(IOU_data_root)
    print("slice_labels : ", slice_labels)
    ordered_labels = []
    iou_rates_dict = {}
    for i in range(len(slice_labels)):
        s_label = "slice_" + str(i)
        ordered_labels.append(s_label)
        iou_rates_dict[s_label] = []
    
    # iou_rates_dict = {}

    invalid_labels = []
    for slice_file in ordered_labels:
        csv_file1 = os.path.join(IOU_data_root, slice_file + "\IOU_rate.csv")
        if not os.path.exists(csv_file1) :
            invalid_labels.append(slice_file)
            continue
        print(csv_file1)
        if os.path.getsize(csv_file1) == 0:
            invalid_labels.append(slice_file)
            continue
        df1 = pd.read_csv(csv_file1, header=None, sep=' ') 
        df1[~(df1 == 0).all(axis=1)]
        
        if df1.empty:
            invalid_labels.append(slice_file)
            continue 
        print("read csv succeed! ")
        df_list = df1[1].values.tolist()
        iou_rates_dict[slice_file] = df_list 

        # print(df1)
        # val_1 = df1.mean()
        # iou_rates.append(val_1[1])
        # valid_labels.append(slice_file)
        # print(iou_rates_dict)
    return iou_rates_dict

       

def plot_single_bar(valid_labels, iou_rates, plot_name):
    x = np.arange(len(valid_labels))  # the label locations
    width = 0.5  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    # for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + width, iou_rates, width)
    bar_labels = [str(round(val, 3)) for val in iou_rates] 
    ax.bar_label(rects, labels=bar_labels, padding=0)
    multiplier += 1
    fig.set_figheight(8)
    fig.set_figwidth(18)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('IOU rate')
    ax.set_title('Mesh Contour IOU rate average')
    ax.set_xticks(x + width, valid_labels)
    plt.xticks(rotation=30, ha='right')
    # ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, 1)
    plt.savefig(plot_name)

def plot_single_bar_with_deviation(valid_labels, iou_rates, deviation, plot_name):
    x = np.arange(len(valid_labels))  # the label locations
    width = 0.5  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    # for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + width, iou_rates, width)
    bar_labels = [str(round(val, 3)) for val in iou_rates] 
    ax.bar_label(rects, labels=bar_labels, padding=0)
    plt.errorbar(x + width, iou_rates, deviation, fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)

    multiplier += 1
    fig.set_figheight(8)
    fig.set_figwidth(18)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('IOU rate')
    ax.set_title('Mesh Contour IOU Rate Average -- Contour Points Sample Distance 7 Pixels')
    ax.set_xticks(x + width, valid_labels)
    plt.xticks(rotation=30, ha='right')
    # ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, 1.2)
    plt.savefig(plot_name)

def plot_all(sample_dist):
    data_root = "D:\data\cycleGrouping"
    file_labels = os.listdir(data_root)
    # sample_dist = 7
    plot_dir = "D:\data\plot_" + str(sample_dist)
    data_dir_name = "sample_dist" + str(sample_dist)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for sub_dir in file_labels:
        # if sub_dir != '6127L': continue 
        IOU_data_root = os.path.join(data_root, sub_dir, "mesh_evaluation", data_dir_name)
        print("IOU_data_root : ", IOU_data_root)
        valid_labels, valid_IOU_rates = get_IOU_rate(IOU_data_root)
        print(valid_labels)
        plot_name = 'mesh_evaluation_all_' + sub_dir + '_.png'
        plot_name = os.path.join(plot_dir, plot_name)
        plot_single_bar(valid_labels, valid_IOU_rates, plot_name)

def get_IOU_rates_all(data_root, sample_dist):
    file_labels = os.listdir(data_root)
    IOU_rates_all = {}
    iou_labels = []
    plot_dir = "D:\data\plot_" + str(sample_dist)
    data_dir_name = "sample_dist" + str(sample_dist)
    for i in range(20):
        key = 'slice_' + str(i)
        IOU_rates_all[key] = []
        iou_labels.append(key)
    for sub_dir in file_labels:
        IOU_data_root = os.path.join(data_root, sub_dir, "mesh_evaluation", data_dir_name)
        print("IOU_data_root : ", IOU_data_root)
        iou_rate_dicts = get_IOU_rate_dict(IOU_data_root)
        for ele in iou_rate_dicts:
            IOU_rates_all[ele] += iou_rate_dicts[ele]
            # print(ele)
    # print(IOU_rates_all) 
    IOU_rates_avg = []
    IOU_rates_deviation = []

    for key in iou_labels:
        data_list = IOU_rates_all[key]
        if len(data_list) == 0:
            IOU_rates_avg.append(0)
            IOU_rates_deviation.append(0)
            continue
        rate_array = np.array(data_list)
        IOU_rates_avg.append(np.mean(rate_array))
        IOU_rates_deviation.append(np.std(rate_array))
    print(IOU_rates_avg)
    print(IOU_rates_deviation)
    # plot_dir = "D:\data\plot_7"
    plot_name = 'mesh_evaluation_all_in_one.png'
    plot_name = os.path.join(plot_dir, plot_name)
    plot_single_bar_with_deviation(iou_labels,IOU_rates_avg, IOU_rates_deviation, plot_name)

    
        # valid_labels, valid_IOU_rates = get_IOU_rate(IOU_data_root)
        # for id in range(len(valid_labels)):
            # IOU_rates_dict[valid_labels[id]].append(valid_IOU_rates[id])
    # print(IOU_rates_dict)

def plot_all_in_one_with_deviations():
    data_root = "D:\data\cycleGrouping"
    sample_dist = 7
    get_IOU_rates_all(data_root, sample_dist)

if __name__ == "__main__":
    sample_dist = 7
    # plot_all(sample_dist)
    plot_all_in_one_with_deviations()