import cv2
import numpy as np
import os


class ContourPointsGenerator:
    def __init__(self) -> None:
        self.dense_contour_ = True
        self.save_imgs=False
        self.work_dir_ = r""
        self.img_stacks_name_ = ""
        self.img_stack_path_ = ""
        self.p_xs_ = []
        self.p_ys_ = []
        self.p_zs_ = []
        self.img_stacks_ = []
        self.dense_contour_stacks_ = []
        self.cv_contour_stacks_ = []
        
        self.sample_step_ = 0
        self.width_ = 398
        self.height_ = 550
        self.save_dense_contour_imgs_ = True
        self.save_cv_contour_imgs_ = True 

    def set_work_dir(self):
        
        self.img_stack_path_ = os.path.join(self.work_dir_, self.img_stacks_name_)
        self.cvimg_save_dir_ = os.path.join(self.work_dir_, "cv_contour_imgs")
        if not os.path.exists(self.cvimg_save_dir_):
            os.mkdir(self.cvimg_save_dir_)
            print("Folder %s created!" % self.cvimg_save_dir_)
        else:
            print("Folder %s already exists" % self.cvimg_save_dir_)

        self.dsimg_save_dir_ = os.path.join(self.work_dir_, "dense_contour_imgs")
        if not os.path.exists(self.dsimg_save_dir_):
            os.mkdir(self.dsimg_save_dir_)
            print("Folder %s created!" % self.dsimg_save_dir_)
        else:
            print("Folder %s already exists" % self.dsimg_save_dir_)
        self.points_save_dir_ = os.path.join(self.work_dir_, "contour_points")
        if not os.path.exists(self.points_save_dir_):
            os.mkdir(self.points_save_dir_)
            print("Folder %s created!" % self.points_save_dir_)
        else:
            print("Folder %s already exists" % self.points_save_dir_)


    def get_dense_contour_points(self,img):
        point_coords = np.transpose(np.nonzero(img))
        img_h, img_w = img.shape
        boundry_points = []
        for p in point_coords:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    img_r = min(max(0, p[0] + i), img_h)
                    img_c = min(max(0, p[1] + j), img_w)
                    if img[img_r][img_c] == 0:
                        boundry_points.append(p)
        return boundry_points
    
    def read_img_stacks(self):
        images = []
        ret, images = cv2.imreadmulti(self.img_stack_path_, images,cv2.IMREAD_ANYDEPTH)
        if len(images) > 1:
            print("Read image stacks successful, img num: ",len(images) )
            self.height_, self.width_ = images[0].shape
            for img in images:
                im_mask = np.ma.masked_greater(img, 0)
                mask = np.ma.getmask(im_mask)
                mask = np.uint8(mask) * 255
                self.img_stacks_.append(mask)
            
    def get_contour_points_cv(self):    
        for img in self.img_stacks_:
            points = []
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for p in contours[0]:
                points.append(p[0])  
            self.cv_contour_stacks_.append(points)
            

    def get_contour_points_dense(self):
        for img in self.img_stacks_:
            points = self.get_dense_contour_points(img)
            self.dense_contour_stacks_.append(points)
    
    def save_contour_imgs(self):
        if self.save_cv_contour_imgs_:
            for i in range(len(self.cv_contour_stacks_)):
                contours = self.cv_contour_stacks_[i]
                image_black = np.zeros((self.height_, self.width_, 3), dtype=np.uint8)
                for p in contours:
                    image_black[p[1]][p[0]] = (255, 0, 0)
                save_path = os.path.join(self.cvimg_save_dir_, str(i) + ".png")
                cv2.imwrite(save_path, image_black)
            print("Save cv contour iamges Finished!" )
        if self.save_dense_contour_imgs_:
            for i in range(len(self.dense_contour_stacks_)):
                contours = self.dense_contour_stacks_[i]
                image_black = np.zeros((self.height_, self.width_, 3), dtype=np.uint8)
                for p in contours:
                    image_black[p[0]][p[1]] = (255, 0, 0)        
                save_path = os.path.join(self.dsimg_save_dir_, str(i) + ".png")
                cv2.imwrite(save_path, image_black)
            print("Save dense contour iamges Finished!" )
    
    def save_cv_contour_points(self):
        save_path = os.path.join(self.points_save_dir_, "cv_contour_" + str(self.sample_step_) + ".xyz" )
        i = 0
        with open(save_path, 'w') as f:    
            while i < len(self.cv_contour_stacks_):
                for p in self.cv_contour_stacks_[i]:
                    f.write(str(p[0]) + " ")
                    f.write(str(p[1]) + " ")
                    f.write(str(i))
                    f.write('\n')
                i += self.sample_step_ + 1
        print("Successfully save cv contour 3D points to file: ", save_path)

    def save_dense_contour_points(self):
        save_path = os.path.join(self.points_save_dir_, "dense_contour_" + str(self.sample_step_) + ".xyz" )
        i = 0
        with open(save_path, 'w') as f:    
            while i < len(self.dense_contour_stacks_):
                for p in self.dense_contour_stacks_[i]:
                    f.write(str(p[1]) + " ")
                    f.write(str(p[0]) + " ")
                    f.write(str(i))
                    f.write('\n')
                i += self.sample_step_ + 1
        print("Successfully save dense contour 3D points to file: ", save_path)
 
    def __call__(self, work_dir, img_stacks_name):
        # Path to the tiff file
        self.work_dir_ = work_dir
        self.img_stacks_name_ = img_stacks_name
        self.set_work_dir()
        print("Set work dir Finished!" )
        self.read_img_stacks()
        print("Read img stacks Finished!" )
        self.get_contour_points_cv()
        print("Get opencv contour points Finished!" )
        self.get_contour_points_dense()
        print("Get dense contour points Finished!" )
        self.save_contour_imgs()
        print("Save contour iamges Finished!" )
        max_sample_steps = 20
        for sample_step in range(max_sample_steps):
            self.sample_step_ = sample_step
            self.save_cv_contour_points()
            self.save_dense_contour_points()


if __name__ == "__main__":
    ptGen = ContourPointsGenerator()
    work_dir = "6007"
    file_name = "6007_segmentation_midsaggital.tif"
    # ptGen(work_dir, file_name)

    work_dir = "C054L"
    file_name = "C054L_segmentation_midsaggital.tif"
    ptGen(work_dir, file_name)
