from PyQt6.QtWidgets import (QWidget, QPushButton, QLineEdit,
        QInputDialog, QApplication, QLabel)
import sys
import os
import cv2 
import numpy as np
import mrcfile
from pathlib import Path
import shutil

def get_img_data(path):
    images = []
    ret, images = cv2.imreadmulti(path, images, cv2.IMREAD_ANYDEPTH)
    img_stacks = []
    if len(images) > 1:
        for img in images:
            # print(img)
            # im_mask = np.ma.masked_greater(img, 0.0)
            # mask = np.ma.getmask(im_mask)
            # new_img = np.float32(mask) * 1.0
            new_img = img.astype(np.float32)
            img_stacks.append(new_img)
        mrc_data = np.stack(img_stacks)
        print(mrc_data.shape)
        return mrc_data

def save_mrc_file(data, out_path):
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(data)    

def convert_img_stacks_to_mrc(in_path, out_path):
    # mrcfile.validate(in_path)
    mrc_data = get_img_data(in_path)
    if mrc_data is None: return
    # if mrc_data == None: return
    save_mrc_file(mrc_data, out_path)

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.init_convert_img_stacks()
        self.init_cal_ma()

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('ET Data Conversion')
        self.show()

    def init_convert_img_stacks(self):
        self.label_in = QLabel(self)
        self.label_in.move(20, 20)
        self.label_in.setText("Input Directory")

        self.label_in = QLabel(self)
        self.label_in.move(20, 70)
        self.label_in.setText("Output Directory")

        self.btn_mrc_generator = QPushButton('Convert Images to MRC', self)
        self.btn_mrc_generator.move(500, 70)
        self.btn_mrc_generator.clicked.connect(self.ConvertImageStacksToMRCFiles)

        self.le_in = QLineEdit(self)
        self.le_in.setFixedWidth(300)
        self.le_in.move(150, 20)

        self.le_out = QLineEdit(self)
        self.le_out.setFixedWidth(300)
        self.le_out.move(150, 70)

    def init_cal_ma(self):
        self.voxel_core_root = QLabel(self)
        self.voxel_core_root.move(20, 160)
        self.voxel_core_root.setText("VoxelCore Directory")

        self.mrc_in_directory = QLabel(self)
        self.mrc_in_directory.move(20, 210)
        self.mrc_in_directory.setText("MRC File Directory")

        self.ma_out_directory = QLabel(self)
        self.ma_out_directory.move(20, 260)
        self.ma_out_directory.setText("MA Output Directory")

        self.le_vc_dir = QLineEdit(self)
        self.le_vc_dir.setFixedWidth(300)
        self.le_vc_dir.move(150, 160)

        self.le_mrc_dir = QLineEdit(self)
        self.le_mrc_dir.setFixedWidth(300)
        self.le_mrc_dir.move(150, 210)

        self.le_ma_dir = QLineEdit(self)
        self.le_ma_dir.setFixedWidth(300)
        self.le_ma_dir.move(150, 260)

        self.tt_value = QLabel(self)
        self.tt_value.move(480, 160)
        self.tt_value.setText("Prune value")

        self.le_tt = QLineEdit(self)
        self.le_tt.setFixedWidth(50)
        self.le_tt.move(560, 160)

        self.btn_ma_generator = QPushButton('Calculate Medial Axes', self)
        self.btn_ma_generator.move(500, 230)
        self.btn_ma_generator.clicked.connect(self.calculate_medial_axes)

        self.lbl_out_mesh_dir = QLabel(self)
        self.lbl_out_mesh_dir.move(20, 310)
        self.lbl_out_mesh_dir.setText("Shape Out Directory")

        self.le_shape_out = QLineEdit(self)
        self.le_shape_out.setFixedWidth(300)
        self.le_shape_out.move(150, 310)

        self.btn_off_generator = QPushButton('Calculate Origin Shape', self)
        self.btn_off_generator.move(500, 310)
        self.btn_off_generator.clicked.connect(self.calculate_medial_axes)


    

    def showDialogIn(self):

        text, ok = QInputDialog.getText(self, 'Image Stacks Data',
                                        'Enter input data directory:')

        if ok:
            self.le_in.setText(str(text))

    def showDialogOut(self):

        text, ok = QInputDialog.getText(self, 'Set mrc output directory',
                                        'Enter output data directory:')

        if ok:
            self.le_out.setText(str(text))

    def ConvertImageStacksToMRCFiles(self):
        input_path = self.le_in.text()
        output_dir = self.le_out.text()
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if os.path.isdir(input_path):
            input_img_files = os.listdir(input_path)
            for stack_name in input_img_files:
                out_name = stack_name.split('.')[0] +  '.mrc'
                output_file = os.path.join(output_dir, out_name)
                input_file = os.path.join(input_path, stack_name)
                try: 
                    convert_img_stacks_to_mrc(input_file, output_file)
                    print("save mrc file to : ", output_file)
                except:
                    print(" convert file %s failed !" %(input_file)) 
            print("Convert image stacks to mrc file finished!")
            print("Convert file number : ", len(input_img_files)) 

    def calculate_medial_axes(self):
        
        # voxel_core_exe = os.path.join(self.le_vc_dir.text(), 'voxelcore.exe')
        voxel_core_dir = os.path.join(self.le_vc_dir.text())
        if not os.path.exists(voxel_core_dir):
            return
        voxel_core_exe =  'voxelcore.exe'
        vc_mode = " -md=vol2ma "
        
        in_mrc_dir = os.path.join(self.le_mrc_dir.text())
        if not os.path.exists(in_mrc_dir):
            return
        ma_out_dir = os.path.join(self.le_ma_dir.text())
        tt_val = self.le_tt.text()
        mrc_file_names = os.listdir(in_mrc_dir)
        for mrc_name in mrc_file_names:
            mrc_file_path = os.path.join(in_mrc_dir, mrc_name)
            
            if os.path.exists(mrc_file_path):
                pure_name = mrc_name.split('.') [0]
                out_ma_name = pure_name + ".ply"
                
                sys_cmd = voxel_core_exe + vc_mode + " " + mrc_file_path + " " + out_ma_name + " -tt " + tt_val
                try:
                    os.chdir(voxel_core_dir)
                    os.system(sys_cmd)
                    out_ma_real_name =  pure_name + "_thinned"+ tt_val + ".ply"
                    out_ma_real_path = os.path.join(voxel_core_dir, out_ma_real_name)
                    if os.path.exists(out_ma_real_path):
                        new_ma_dest = os.path.join(ma_out_dir, out_ma_real_name)
                        shutil.move(out_ma_real_path, new_ma_dest)

                    out_r_real_name = pure_name + "_thinned"+ tt_val + ".r"
                    out_r_real_path = os.path.join(voxel_core_dir, out_r_real_name)
                    if os.path.exists(out_r_real_path):
                        new_r_dest = os.path.join(ma_out_dir, out_r_real_name)
                        shutil.move(out_r_real_path, new_r_dest)
                    
                except:
                    print( " Command excuted failed : ", sys_cmd)

        print("MA calculation finished!!!")

    def calculate_original_shape(self):
        
        # voxel_core_exe = os.path.join(self.le_vc_dir.text(), 'voxelcore.exe')
        voxel_core_dir = os.path.join(self.le_vc_dir.text())
        if not os.path.exists(voxel_core_dir):
            return
        voxel_core_exe =  'voxelcore.exe'
        vc_mode = " -md=vol2mesh "
        
        in_mrc_dir = os.path.join(self.le_mrc_dir.text())
        if not os.path.exists(in_mrc_dir):
            return
        shape_out_dir = os.path.join(self.le_shape_out.text())
        tt_val = self.le_tt.text()
        mrc_file_names = os.listdir(in_mrc_dir)
        for mrc_name in mrc_file_names:
            mrc_file_path = os.path.join(in_mrc_dir, mrc_name)
            
            if os.path.exists(mrc_file_path):
                pure_name = mrc_name.split('.') [0]
                # out_ma_name = pure_name + 
                
                sys_cmd = voxel_core_exe + vc_mode + " " + mrc_file_path + " " + pure_name + "  -onlyBndryVts false"
                try:
                    os.chdir(voxel_core_dir)
                    os.system(sys_cmd)
                    out_shape_real_name =  pure_name + ".off"
                    out_shape_real_path = os.path.join(voxel_core_dir, out_shape_real_name)
                    if os.path.exists(out_shape_real_path):
                        new_shape_dest = os.path.join(shape_out_dir, out_shape_real_name)
                        shutil.move(out_shape_real_path, new_shape_dest)
                except:
                    print( " Command excuted failed : ", sys_cmd)

        print("original shape calculation finished!!!")

def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
