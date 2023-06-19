import cv2
import numpy as np
import os

def get_contour_points(img_stack_path, sample_step=0, save_dir ="", save_imgs=False):
    # List to store the loaded image
    images = []
    ret, images = cv2.imreadmulti(img_stack_path, images,cv2.IMREAD_ANYDEPTH)
    print("tiff img counts ", len(images))
    x = []
    y = []
    z = []
    if len(images) > 1:
        i = 0
        while i < len(images):
            # Dynamic name
            name = 'Image'+str(i)
            im_mask = np.ma.masked_greater(images[i], 0)
            im_mask = np.ma.getmask(im_mask)
            img2 = np.uint8(im_mask) * 255
            
            contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_black = np.zeros((im_mask.shape[0], im_mask.shape[1],3), dtype=np.uint8)
            
            for p in contours[0]:
                image_black[p[0][1]][p[0][0]] = (255, 0, 0)
                x.append(p[0][0])
                y.append(p[0][1])
                z.append(i)

            if save_imgs:
                out_name = str(i) + ".png"
                out_path = os.path.join(save_dir, out_name)  
                cv2.imwrite(out_path, image_black) 
            i += sample_step + 1
                # 
    return (x, y, z)

def save_contour_points(contour_points, save_path):
    p_xs = contour_points[0]
    p_ys = contour_points[1]
    p_zs = contour_points[2]
    with open(save_path, 'w') as f:
        for i in range(len(p_xs)):
            f.write(str(p_xs[i]) + " ")
            f.write(str(p_ys[i]) + " ")
            f.write(str(p_zs[i]))
            f.write('\n')
    

def generate_contour_points_from_imgs():
    # Path to the tiff file
    img_stack_path = r"D:\projects\Files\data\6007\6007_segmentation_midsaggital.tif"
    img_save_dir = r"D:\projects\IMG\contour_imgs"
    points_save_dir = r"D:\projects\IMG\contour_points"

    save_contour_imgs = False
    max_sample_steps = 10
    for sample_step in range(max_sample_steps):
        coordinates = get_contour_points(img_stack_path, sample_step, img_save_dir, save_contour_imgs)
        file_name = "contour_points" + str(sample_step) + ".xyz"
        contour_points_dir = os.path.join(points_save_dir, file_name)
        save_contour_points(coordinates, contour_points_dir)

if __name__ == "__main__":
    generate_contour_points_from_imgs()