import cv2
import numpy as np
import os
import kdtree
import math
import copy


def dist_between_p( p0, p1):
    dx = p0[0] - p1[0]
    dy = p0[1] - p1[1]
    return math.sqrt(dx*dx + dy*dy)

class ContourPointsGenerator:
    def __init__(self) -> None:
        self.dense_contour_ = True
        self.save_imgs=False
        self.work_dir_ = r""
        self.img_stacks_name_ = ""
        self.img_stack_path_ = ""
        self.ori_img_stack_path_ = ''

        self.img_stacks_ = []
        self.dense_contour_stacks_ = []
        self.cv_contour_stacks_ = []        
        self.sample_step_ = 0
        self.width_ = 398
        self.height_ = 550
        self.save_dense_contour_imgs_ = True
        self.save_cv_contour_imgs_ = True 
        self.save_opt_cv_contour_imgs_ = True
        self.img_scale_ = 1
        self.img_resize_ = False
        self.sample_dense_points_ = True
        self.kd_dist_thred_ = 5
        self.sample_contour_stacks_ = []
        self.opt_cv_contour_stacks_ = []

    def dist_3d(self, p1, p2):
        dist_sum = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        return math.sqrt(dist_sum)

        
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
            resize_dim = (self.width_ // self.img_scale_ , self.height_ // self.img_scale_)
            for img in images:
                if self.img_resize_: 
                    img = cv2.resize(img, resize_dim, interpolation= cv2.INTER_LINEAR)
                # print("----------------- resize img dim ", resize_img.shape)
                
                im_mask = np.ma.masked_greater(img, 0.5)
                mask = np.ma.getmask(im_mask)
                mask = np.uint8(mask) * 255

                # print("----------------- mask img dim ", mask.shape)
                self.img_stacks_.append(mask)

    def read_ori_img_stacks(self):
        images = []
        self.origin_img_stacks_ = []
        print("read origin img : ", self.ori_img_stack_path_)
        ret, images = cv2.imreadmulti(self.ori_img_stack_path_, images,cv2.IMREAD_ANYDEPTH)
        self.origin_img_stacks_ = images
        # if len(images) > 1:
        #     print("Read image stacks successful, img num: ",len(images) )
        #     self.height_, self.width_ = images[0].shape
        #     resize_dim = (self.width_ // self.img_scale_ , self.height_ // self.img_scale_)
        #     for img in images:
        #         if self.img_resize_: 
        #             img = cv2.resize(img, resize_dim, interpolation= cv2.INTER_LINEAR)
        #         # print("----------------- resize img dim ", resize_img.shape)
                
        #         im_mask = np.ma.masked_greater(img, 0.5)
        #         mask = np.ma.getmask(im_mask)
        #         mask = np.uint8(mask) * 255

        #         # print("----------------- mask img dim ", mask.shape)
        #         self.img_stacks_.append(mask)
            
    def get_contour_points_cv(self):    
        for img in self.img_stacks_:
            points = []
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_contour_id = 0
            max_contour_size = 0
            for i in range(len(contours)):
                if len(contours[i]) > max_contour_size:
                    max_contour_size = len(contours[i])
                    max_contour_id = i
            for p in contours[max_contour_id]:
                points.append(p[0])  
            self.cv_contour_stacks_.append(points)
            

    def get_contour_points_dense(self):
        for img in self.img_stacks_:
            points = self.get_dense_contour_points(img)
            self.dense_contour_stacks_.append(points)

    def sample_dense_points_with_kdtree(self):
        if len(self.dense_contour_stacks_) == 0:
            return
        i = 0
        while i < len(self.dense_contour_stacks_):
            contour_pts = []
            my_kdTree = kdtree.create(dimensions=3)
            for p in self.dense_contour_stacks_[i]:
                new_p = (p[1], p[0], i)
                if self.img_resize_ :
                    new_p = (p[1] * self.img_scale_, p[0] * self.img_scale_, i)
                nearest_pts = my_kdTree.search_nn( new_p )
                
                if nearest_pts == None :
                    my_kdTree.add(new_p)
                    contour_pts.append(new_p)
                    continue

                if nearest_pts[0].dist(new_p) < self.kd_dist_thred_:  
                    continue
                else :
                    my_kdTree.add(new_p)
                    contour_pts.append(new_p)
            i +=  1
            self.sample_contour_stacks_.append(contour_pts)
    
    def save_contour_imgs(self):
        img_shape = (self.height_ , self.width_ , 3)
        if self.img_resize_ :
            img_shape = (self.height_ // self.img_scale_, self.width_ // self.img_scale_, 3)
        if self.save_cv_contour_imgs_:
            
            for i in range(len(self.cv_contour_stacks_)):
                contours = self.cv_contour_stacks_[i]
                image_black = np.zeros(img_shape, dtype=np.uint8)
                for p in contours:
                    if p[1] >= img_shape[0] and p[0] >= img_shape[1] :
                        continue
                    image_black[p[0]][p[1]] = (255, 0, 0)
                save_path = os.path.join(self.cvimg_save_dir_, str(i) + ".png")
                cv2.imwrite(save_path, image_black)
            print("Save cv contour iamges Finished!" )
       
        if self.save_dense_contour_imgs_:
            print("Save dense contour iamges count ",  len(self.dense_contour_stacks_))
            for i in range(len(self.dense_contour_stacks_)):
                contours = self.dense_contour_stacks_[i]
                image_black = np.zeros(img_shape, dtype=np.uint8)
                for p in contours:
                    if p[1] >= img_shape[1] and p[0] >= img_shape[0] :
                        continue
                    image_black[p[1]][p[0]] = (255, 0, 0)        
                save_path = os.path.join(self.dsimg_save_dir_, str(i) + ".png")
                cv2.imwrite(save_path, image_black)
            print("Save dense contour iamges Finished!" )

    def save_opt_cv_contour_imgs(self):
        img_shape = (self.height_ , self.width_ , 3)
        out_path = os.path.join(self.cvimg_save_dir_, "opt_cv_contour_imgs_ " + str(self.kd_dist_thred_) )
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if self.save_opt_cv_contour_imgs_:
            print("contour count : ", len(self.opt_cv_contour_stacks_))
            for c_i in range(len(self.opt_cv_contour_stacks_)):
                contours = self.opt_cv_contour_stacks_[c_i]
                origin_img = copy.deepcopy(self.img_stacks_[c_i])
                origin_img = cv2.cvtColor(origin_img,cv2.COLOR_GRAY2RGB)
                # image_black = np.zeros(img_shape, dtype=np.uint8)
                for p in contours:
                    if p[1] >= img_shape[1] and p[0] >= img_shape[0] :
                        continue
                    cv2.circle(origin_img,(int(p[0] + 0.5), int(p[1] + 0.5)), 1, (255,0,0), -1)

                for i in range(len(contours)-1):
                    next_id = (i + 1) % len(contours)
                    s_p = (int(contours[i][0] + 0.5), int(contours[i][1] + 0.5))
                    e_p = (int(contours[next_id][0]+ 0.5), int(contours[next_id][1]+ 0.5))
                    cv2.line(origin_img, s_p, e_p, (0,0,255), 1)
                    
                    # image_black[p[1]][p[0]] = (255, 0, 0)
                save_path = os.path.join(out_path, str(c_i) +"_contour_dist_" + str(self.kd_dist_thred_) + ".png")
                cv2.imwrite(save_path, origin_img)
            print("Save opt cv contour iamges Finished!" )

    def optimize_cv_contour_points(self):
        self.opt_cv_contour_stacks_ = []
        for img_points in self.cv_contour_stacks_:
            # pt_visit = np.zeros((self.height_, self.width_))
            pt_size = len(img_points)
            opt_pts = []
            pre_id = 0

            pass_dist = 0.0
            for i in range(pt_size):

                if i > pre_id:
                    new_id = i 
                    dx = img_points[new_id][0] - img_points[i-1][0]
                    dy = img_points[new_id][1] - img_points[i-1][1]

                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist + pass_dist < self.kd_dist_thred_:
                        pass_dist += dist
                        continue
                    else :
                        if i - 1 > pre_id:
                            opt_pts.append(img_points[i-1])
                            pre_id = i - 1
                            pass_dist = 0
                        if dist + pass_dist < self.kd_dist_thred_:
                            pass_dist += dist
                            continue
                        ratio = (pass_dist + dist) / self.kd_dist_thred_
                        pass_dist = 0.0
                        if ratio > 1.0:
                            sep_count = int(ratio) + 1
                            x_step = dx / sep_count
                            y_step = dy / sep_count
                            for int_id in range(sep_count):
                                new_x = img_points[pre_id][0] + x_step * (1 + int_id)
                                new_y = img_points[pre_id][1] + y_step * (1 + int_id)
                            opt_pts.append((new_x, new_y))
                        if new_id != 0: 
                            opt_pts.append(img_points[i])
                        pre_id = i
                else :
                    opt_pts.append(img_points[i])
            if len(img_points) > 0:
                end_p = img_points[-1]
                sta_p = img_points[0]
                dx = sta_p[0] - end_p[0]
                dy = sta_p[1] - end_p[1]
                dist = math.sqrt(dx * dx + dy * dy)

                for i in range(3):
                    newx = end_p[0] + 1/3.0 * (1 + i) * dx
                    newy = end_p[1] + 1/3.0 * (1 + i) * dy
                    opt_pts.append((newx, newy))

            self.opt_cv_contour_stacks_.append(opt_pts)


    
    def optimize_cv_contour_points2(self):
        self.opt_cv_contour_stacks_ = []
        for contour_id in range(len(self.cv_contour_stacks_)):
            # if contour_id != 40:
            #     continue
            img_points = self.cv_contour_stacks_[contour_id]
        # for img_points in self.cv_contour_stacks_:
            # pt_visit = np.zeros((self.height_, self.width_))
            pt_size = len(img_points)
            opt_pts = []
            
            re_dist = 0
            
            if len(img_points) == 0:
                continue
            p_start = img_points[0]
            pre_p = img_points[0]
            for i in range(pt_size + 1):
                p_id = i 
                if i == pt_size:
                    p_id = 0
                # print ("p_id ", p_id )
                # print ("re_dist ", re_dist)
                cur_p = img_points[p_id]
                if len(opt_pts) == 0:
                    opt_pts.append(cur_p)
                else :
                    # pre_p = opt_pts[-1]
                    dx = cur_p[0] - pre_p[0]
                    dy = cur_p[1] - pre_p[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    
                    dx = dx / dist
                    dy = dy / dist
                    # print("dist : %f, dx : %f,  dy: %f "%(dist, dx, dy))
                    forward_dist = self.kd_dist_thred_ - re_dist
                    # print ("forward_dist ", forward_dist)
                    new_re_dist = dist - forward_dist
                    # print ("forward_dist: %f, new_re_dist : %f"%(forward_dist,new_re_dist))
                    if new_re_dist >= 0:
                        new_x = pre_p[0] + dx * forward_dist
                        new_y = pre_p[1] + dy * forward_dist
                        if i == pt_size:
                            dx_s = p_start[0] - new_x
                            dy_s = p_start[1] - new_y
                            dist_s = math.sqrt(dx_s*dx_s + dy_s* dy_s)
                            if dist_s < self.kd_dist_thred_* 0.5: 
                                break
                        opt_pts.append((new_x, new_y))
                        # print ("add new p : %f, %f, dist from pre p: %f"%( new_x, new_y, dist_between_p(opt_pts[-1], opt_pts[-2])))
                        pre_p = (new_x, new_y)
                        while new_re_dist >= self.kd_dist_thred_:
                            new_x = pre_p[0] + dx * self.kd_dist_thred_
                            new_y = pre_p[1] + dy * self.kd_dist_thred_
                            if i == pt_size:
                                dx_s = p_start[0] - new_x
                                dy_s = p_start[1] - new_y
                                dist_s = math.sqrt(dx_s*dx_s + dy_s* dy_s)
                                if dist_s < self.kd_dist_thred_* 0.5:
                                    # print("new p is too close to the start point --------------------") 
                                    break
                            opt_pts.append((new_x, new_y))
                            # print ("add new p : %f, %f "%( new_x, new_y))
                            # print ("add new p : %f, %f, dist from pre p: %f"%( new_x, new_y, dist_between_p(opt_pts[-1], opt_pts[-2])))
                            pre_p = (new_x, new_y)
                            new_re_dist -= self.kd_dist_thred_
                        re_dist = new_re_dist
                        # print ("re_dist 1  ", re_dist)
                    else :
                        re_dist += dist
                    pre_p = img_points[p_id]
                        # print ("re_dist 2 ", re_dist)

                    # print("pre p : %f , %f " %(pre_p[0], pre_p[1]))

            # for i in range(len(opt_pts)):
            #     next_id = (i + 1) % len(opt_pts)
            #     dx = img_points[next_id][0] - img_points[i][0]
            #     dy = img_points[next_id][1] - img_points[i][1]
            #     dist = math.sqrt(dx * dx + dy * dy)
            #     print(" dist ", i, " ",   dist)

            self.opt_cv_contour_stacks_.append(opt_pts)

    def cal_contour_points_tangent_and_normals(self):
        self.contour_tangent_stacks_ = []
        self.contour_normal_stacks_ = []

        p_normal = (0, 0, 1)
        for i in range(len(self.opt_cv_contour_stacks_)):
            cur_contour = self.opt_cv_contour_stacks_[i]
            cur_tangent_list = []
            cur_normal_list  = []

            for j in range(len(cur_contour)):
                pre_id = (j - 1) % len(cur_contour)
                aft_id = (j + 1) % len(cur_contour)
                # print("id : ", pre_id, aft_id)
                tangent_x = cur_contour[aft_id][0] - cur_contour[pre_id][0]
                tangent_y = cur_contour[aft_id][1] - cur_contour[pre_id][1]
                # print("tangent : ", tangent_x, tangent_y)
                cur_tangent = (tangent_x, tangent_y, 0)
                cur_tangent_list.append(cur_tangent)
                cur_normal = np.cross(p_normal, cur_tangent)
                cur_normal = cur_normal / np.sqrt(cur_normal.dot(cur_normal))
                cur_normal_list.append(cur_normal)
            self.contour_normal_stacks_.append(cur_normal_list)
            self.contour_tangent_stacks_.append(cur_tangent_list)
    
    def CalGradient(self):
        self.contour_gradientX_stacks_ = []
        self.contour_gradientY_stacks_ = []
        self.contour_gradientZ_stacks_ = []
        img_num = len(self.origin_img_stacks_)
        for i in range(img_num):
            pre_id = max(0, i -1)
            aft_id = min(img_num, i + 1)
            gradient_z = (self.origin_img_stacks_[aft_id] - self.origin_img_stacks_[pre_id])/2
            self.contour_gradientZ_stacks_.append(gradient_z)

            cur_img = self.origin_img_stacks_[i]
            row, col = cur_img.shape
            left_x = np.zeros_like(cur_img)
            left_x[:,1:] = cur_img[:, :col-1]
            left_x[:, 0] = cur_img[:, 0]

            right_x = np.zeros_like(cur_img)
            right_x[:,:col-1] = cur_img[:, 1:]
            right_x[:, col-1] = cur_img[:, col-1]
            gradient_x = (right_x - left_x) / 2.0
            self.contour_gradientX_stacks_.append(gradient_x)
            
            up_y = np.zeros_like(cur_img)
            up_y[1:, :] = cur_img[:row-1, :]
            up_y[0,  :] = cur_img[0, :]

            dw_y = np.zeros_like(cur_img)
            dw_y[: row-1, :] = cur_img[1:,:]
            dw_y[:, row-1]   = cur_img[row-1, :]
            gradient_y = (dw_y - up_y) / 2.0
            self.contour_gradientY_stacks_.append(gradient_y)
            
    def save_cv_contour_points(self):
        save_path = os.path.join(self.points_save_dir_, "cv_contour_" + str(self.sample_step_) + ".xyz" )
        i = 0
        with open(save_path, 'w') as f:    
            while i < len(self.cv_contour_stacks_):
                for p in self.cv_contour_stacks_[i]:
                    f.write(str(p[0] * self.img_scale_) + " ")
                    f.write(str(p[1] * self.img_scale_) + " ")
                    f.write(str(i))
                    f.write('\n')
                i += self.sample_step_ + 1
        print("Successfully save cv contour 3D points to file: ", save_path)

    def save_opt_cv_contour_points(self):
        save_dir = os.path.join(self.points_save_dir_, "opt_cv_contour_" + str(self.kd_dist_thred_))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "opt_cv_contour_" + str(self.sample_step_) + ".xyz" )
        i = 0
        with open(save_path, 'w') as f:    
            while i < len(self.opt_cv_contour_stacks_):
                for p in self.opt_cv_contour_stacks_[i]:
                    px = p[0]
                    py = p[1]
                    if self.img_scale_:
                        px = p[0] * self.img_scale_
                        py = p[1] * self.img_scale_
                    f.write(str(px) + " ")
                    f.write(str(py) + " ")
                    f.write(str(i))
                    f.write('\n')
                i += self.sample_step_ + 1
        print("Successfully save cv contour 3D points to file: ", save_path)
    
    def save_opt_cv_contour_tangents(self):
        save_dir = os.path.join(self.points_save_dir_, "contour_tangent_" + str(self.kd_dist_thred_))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "contour_tangent_" + str(self.sample_step_) + ".xyz" )
        i = 0
        with open(save_path, 'w') as f:    
            while i < len(self.contour_tangent_stacks_):
                for p in self.contour_tangent_stacks_[i]:
                    px = p[0]
                    py = p[1]
                    f.write(str(px) + " ")
                    f.write(str(py) + " ")
                    f.write(str(p[2]))
                    f.write('\n')
                i += self.sample_step_ + 1
        print("Successfully save cv contour tangent to file: ", save_path)

    def save_opt_cv_contour_points_and_normals(self):
        save_dir = os.path.join(self.points_save_dir_, "contour_normal_" + str(self.kd_dist_thred_))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "contour_normal_" + str(self.sample_step_) + ".obj")
        i = 0
        with open(save_path, 'w') as f: 
            p_num = 0   
            while i < len(self.opt_cv_contour_stacks_):
                cur_contour_points = self.opt_cv_contour_stacks_[i]
                cur_contour_normals = self.contour_normal_stacks_[i]
                for p_id in range(len(cur_contour_points)):
                    p = cur_contour_points[p_id]
                    pn = cur_contour_normals[p_id]
                    p_num += 1
                    f.write("v ")
                    f.write(str(p[0]) + " ")
                    f.write(str(p[1]) + " ")
                    f.write(str(i))
                    f.write('\n')
                    f.write("v ")
                    f.write(str(p[0] + 3 * pn[0]) + " ")
                    f.write(str(p[1] + 3 * pn[1]) + " ")
                    f.write(str(i + 3 * pn[2]))
                    f.write('\n')
                i += self.sample_step_ + 1
            for id in range(p_num):
                f.write("l ")
                f.write(str(2 *id + 1) + " ")
                f.write(str(2 *id + 2) )
                f.write('\n')
        print("Successfully save cv contour normal to file: ", save_path)

    def save_opt_cv_contour_normals(self):
        save_dir = os.path.join(self.points_save_dir_, "contour_normal_" + str(self.kd_dist_thred_))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "contour_normal_" + str(self.sample_step_) + ".xyz")
        i = 0
        with open(save_path, 'w') as f: 
            p_num = 0   
            while i < len(self.opt_cv_contour_stacks_):
                
                cur_contour_normals = self.contour_normal_stacks_[i]
                for p_id in range(len(cur_contour_normals)):
                    
                    pn = cur_contour_normals[p_id]
                    p_num += 1
                    f.write(str(pn[0]) + " ")
                    f.write(str(pn[1]) + " ")
                    f.write(str(pn[2]))
                    f.write('\n')
                i += self.sample_step_ + 1
        print("Successfully save cv contour normal to file: ", save_path)

    def save_opt_cv_contour_points_gradients(self):
        save_dir = os.path.join(self.points_save_dir_, "contour_gradient_" + str(self.kd_dist_thred_))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "contour_gradient_" + str(self.sample_step_) + ".obj")
        i = 0
        with open(save_path, 'w') as f: 
            p_num = 0   
            while i < len(self.opt_cv_contour_stacks_):
                cur_contour_points = self.opt_cv_contour_stacks_[i]
                cg_x = self.contour_gradientX_stacks_[i]
                cg_y = self.contour_gradientY_stacks_[i]
                cg_z = self.contour_gradientZ_stacks_[i]
                gradients = []
                max_norm = 0
                for p_id in range(len(cur_contour_points)):
                    p = cur_contour_points[p_id]
                    gx = cg_x[int(p[0]), int(p[1])]
                    gy = cg_y[int(p[0]), int(p[1])]
                    gz = cg_z[int(p[0]), int(p[1])]
                    gradients.append((gx, gy, gz))
                    norm = np.sqrt(gx*gx + gy*gy + gz*gz)
                    if max_norm < norm:
                        max_norm = norm

                for p_id in range(len(cur_contour_points)):
                    p = cur_contour_points[p_id]
                    pn = gradients[p_id] 
                    p_num += 1
                    f.write("v ")
                    f.write(str(p[0]) + " ")
                    f.write(str(p[1]) + " ")
                    f.write(str(i))
                    f.write('\n')
                    f.write("v ")
                    f.write(str(p[0] + 10 * pn[0] / max_norm) + " ")
                    f.write(str(p[1] + 10 * pn[1]/ max_norm)  + " ")
                    f.write(str(i + 10 * pn[2]/ max_norm))
                    f.write('\n')
                i += self.sample_step_ + 1
            for id in range(p_num):
                f.write("l ")
                f.write(str(2 *id + 1) + " ")
                f.write(str(2 *id + 2) )
                f.write('\n')
        print("Successfully save cv contour gradient to file: ", save_path)

    def save_cv_contour_points(self):
        save_path = os.path.join(self.points_save_dir_, "cv_contour_" + str(self.sample_step_) + ".xyz" )
        i = 0
        with open(save_path, 'w') as f:    
            while i < len(self.cv_contour_stacks_):
                for p in self.cv_contour_stacks_[i]:
                    f.write(str(p[0] * self.img_scale_) + " ")
                    f.write(str(p[1] * self.img_scale_) + " ")
                    f.write(str(i))
                    f.write('\n')
                i += self.sample_step_ + 1
        print("Successfully save cv contour 3D points to file: ", save_path)

    def save_opt_cv_contour_as_planes(self):
        save_dir = os.path.join(self.points_save_dir_, "opt_cv_contour_plane_" + str(self.kd_dist_thred_))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "opt_cv_contour_plane_" + str(self.sample_step_) + ".contour" )
        i = 0
        coord_scale = 1.0
        with open(save_path, 'w') as f:
            contour_count = (len(self.opt_cv_contour_stacks_) + self.sample_step_) // (self.sample_step_ + 1) 
            f.write(str(contour_count) + "\n")
            while i < len(self.opt_cv_contour_stacks_):
                f.write("0 0 1 -" + str(i/coord_scale) + "\n")
                v_count = len(self.opt_cv_contour_stacks_[i])
                f.write(str(v_count) + " " + str(v_count) + "\n")
                for p in self.opt_cv_contour_stacks_[i]:
                    px = p[0]
                    py = p[1]
                    if self.img_scale_:
                        px = p[0] * self.img_scale_
                        py = p[1] * self.img_scale_
                    f.write(str(px / coord_scale) + " ")
                    f.write(str(py / coord_scale ) + " ")
                    f.write(str(i / coord_scale))
                    f.write('\n')
                for v_id in range(v_count):
                    f.write(str(v_id) + " ")
                    next_id = (v_id + 1) % v_count
                    f.write(str(next_id) + " 0 1")
                    f.write('\n')
                i += self.sample_step_ + 1

        print("Successfully save cv contour 3D points to file: ", save_path)


    def save_dense_contour_points(self):
        save_path = os.path.join(self.points_save_dir_, "dense_contour_" + str(self.sample_step_) + ".xyz" )
        
        i = 0
        with open(save_path, 'w') as f:    
            while i < len(self.dense_contour_stacks_):
                for p in self.dense_contour_stacks_[i]:
                    f.write(str(p[1] * self.img_scale_) + " ")
                    f.write(str(p[0] * self.img_scale_) + " ")
                    f.write(str(i))
                    f.write('\n')
                i += self.sample_step_ + 1
        print("Successfully save dense contour 3D points to file: ", save_path)
    
    def save_sample_contour_points(self):
        save_dir = os.path.join(self.points_save_dir_, "kd_dist_" + str(self.kd_dist_thred_))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "contour_" + str(self.kd_dist_thred_) + "_" + str(self.sample_step_) + ".xyz" )
        i = 0
        with open(save_path, 'w') as f:
            while i < len(self.dense_contour_stacks_): 
                for p in self.sample_contour_stacks_[i]:
                    str_line = str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n"
                    f.write(str_line)
                i += self.sample_step_ + 1
        print("Successfully save dense contour sampled 3D points to file: ", save_path)


    def __call__(self, work_dir, img_stacks_name):
        # Path to the tiff file
        self.work_dir_ = work_dir
        self.img_stacks_name_ = img_stacks_name
        self.set_work_dir()
        print("Set work dir Finished!" )
        self.read_img_stacks()
        print("Read img stacks Finished!" )
        self.read_ori_img_stacks()
        print("Read original img stacks Finished!" )
        self.get_contour_points_cv()
        print("Get opencv contour points Finished!" )
        
        get_dense_contour = False
        if get_dense_contour:
            self.get_contour_points_dense()
            print("Get dense contour points Finished!" )
            if self.sample_dense_points_:
                self.sample_dense_points_with_kdtree()
                print("Sample dense contour points Finished!" )
        # self.save_contour_imgs()
        print("Save contour iamges Finished!" )
        dist_sample_list = [3, 5, 7, 9]
        # dist_sample_list = [8]
        max_sample_steps = 10

        cal_gradient = True
        if cal_gradient:
            self.CalGradient()
        
        for dist_step in dist_sample_list:
            self.kd_dist_thred_ = dist_step 
            self.optimize_cv_contour_points2()
            self.save_opt_cv_contour_imgs()
            self.cal_contour_points_tangent_and_normals()
            
            print("Optimize opencv contour points Finished!" )
            for sample_step in range(max_sample_steps):
                self.sample_step_ = sample_step
                #self.save_cv_contour_points()
                if get_dense_contour:
                    self.save_dense_contour_points()
                    self.save_sample_contour_points()
                
                self.save_opt_cv_contour_points()
                self.save_opt_cv_contour_as_planes()
                self.save_opt_cv_contour_points_and_normals()
                self.save_opt_cv_contour_tangents()

def run_bateches():
    work_dir = "/home/jjxia/Documents/projects/All_Elbow/elbow_data/segmentations_tiffstack"
    origin_img_dir = "/home/jjxia/Documents/projects/All_Elbow/elbow_data/scans_tiffstacks"
    file_names = os.listdir(work_dir)
    out_dir = "/home/jjxia/Documents/projects/All_Elbow/contours_1"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file_name in file_names:
        file_path = os.path.join(work_dir, file_name)
        out_sub_dir = os.path.join(out_dir, file_name.split('_')[0])
        origin_img_name =  file_name.split('_')[0] + "_scan_midsaggital.tif"
        origin_img_path = os.path.join(origin_img_dir, origin_img_name)
        if not os.path.exists(out_sub_dir):
            os.mkdir(out_sub_dir)
        try:
            ptGen = ContourPointsGenerator()
            ptGen.ori_img_stack_path_ = origin_img_path
            ptGen(out_sub_dir, file_path)
        except: 
            print(file_name, " failed!")
        


if __name__ == "__main__":
    # ptGen = ContourPointsGenerator()
    # work_dir = "6007"
    # file_name = "6007_segmentation_midsaggital.tif"
    # ptGen(work_dir, file_name)

    # work_dir = "C054L"
    # file_name = "C054L_segmentation_midsaggital.tif"
    # ptGen(work_dir, file_name)
    run_bateches()