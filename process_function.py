import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
import time

class AppWindow:

    def __init__(self):
        start = ''

    def Background_Subtraction_Func(self):
        def Single_Gaussian_Model(frame_for_model):
            cap = cv2.VideoCapture(r"Q1_Image\bgSub.mp4")
            frame_index = 0
            std_array = []
            mean_array = []
            frame_array = []

            while(cap.isOpened() and frame_index < frame_for_model):
                ret, frame = cap.read()
                frame_index += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flatten_gray = gray.flatten()
                if(frame_index <= 1):
                    flatten_size = np.size(flatten_gray) 
                    frame_array = np.empty((0,flatten_size))
                    frame_array = np.append(frame_array, np.array([flatten_gray]), axis = 0)
                else:    
                    frame_array = np.append(frame_array, np.array([flatten_gray]), axis = 0)

            cap.release()
            cv2.destroyAllWindows()
            frame_array = np.transpose(frame_array)
            row, col = np.shape(frame_array)
            for i in range(row):
                std_value = np.std(frame_array[i])
                if(std_value<5):
                    std_value=5
                std_array = np.append(std_array, std_value)
                
                mean_value = np.mean(frame_array[i])
                mean_array = np.append(mean_array, mean_value)

            return std_array, mean_array

        def Background_Subtraction(input_frame, std_array, mean_array):
            width, height = np.shape(input_frame)
            flatten_frame = input_frame.flatten()
            length = width*height
            frame_result = np.zeros(length)
            
            for i in range(length):
                pixel_calculate = np.abs(flatten_frame[i] - mean_array[i])
                if(pixel_calculate > 5*std_array[i]):
                    frame_result[i] = 255
                else:
                    frame_result[i] = 0
            frame_result = np.reshape(frame_result, (width, height))
            # probability_value += 1/np.sqrt(2*pi*std_array[j])*np.exp(-1/2*(np.square(i-mean_array[j])/np.square(std_array[j])))
            # probability_value = probability_value/50

            return frame_result            

        std_array, mean_array = Single_Gaussian_Model(50)
        cap = cv2.VideoCapture(r"Q1_Image\bgSub.mp4")
        prev_time = 0
        new_time = 0

        while(cap.isOpened()):
            ret, frame = cap.read() #if video read correctly, the return value ret will be true
            if(ret == True):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = Background_Subtraction(gray, std_array, mean_array)

                # change binary result(1 channel) to three channel
                three_channel_image = np.zeros_like(frame)
                three_channel_image[:,:,0] = result
                three_channel_image[:,:,1] = result
                three_channel_image[:,:,2] = result

                new_time = time.time()
                fps = 1/(new_time - prev_time)
                prev_time = new_time

                fps = round(fps, 2)
                fps_str = 'fps:' + str(fps)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 0.5
                font_color = (255, 10, 100)
                line_size = 1
                cv2.putText(three_channel_image, fps_str, (10, 20), font, font_size, font_color, line_size, cv2.LINE_AA)

                both = np.hstack((frame, three_channel_image))
                cv2.imshow('Background Subtraction', both)

                if(cv2.waitKey(1) & 0xFF == ord('q')):
                    break
            else:
                break

        print('totally finish')
        cap.release()
        cv2.destroyAllWindows()

    def Preprocessing_Func(self):
        cap = cv2.VideoCapture(r"Q2_Image\opticalFlow.mp4")
        ret, frame = cap.read()
        # Read image
        #im = cv2.imread("01.JPG")
        
        im=frame
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        # Filter by Area.
        params.filterByArea = True
        #params.maxArea=200
        #params.minArea = 75
        params.maxArea=200
        params.minArea = 30
         
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.75
         
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
            
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
         
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
        	detector = cv2.SimpleBlobDetector(params)
        else : 
        	detector = cv2.SimpleBlobDetector_create(params)
         
         
        # Detect blobs.
        keypoints = detector.detect(im)
         
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob
         
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        
        #繪製十字及方框
        def drawGG(points2f):
            SideLong=5#邊長
            for Center in points2f:
                print(Center)#座標
                print(Center[0],Center[1])
                #橫
                startX=int(Center[0])-SideLong
                startY=int(Center[1])
                endX=int(Center[0])+SideLong
                endY=int(Center[1])
                cv2.line(im_with_keypoints,(startX,startY),(endX,endY), (0, 0, 255), 1)
                #直
                startX=int(Center[0])
                startY=int(Center[1])-SideLong
                endX=int(Center[0])
                endY=int(Center[1])+SideLong
                cv2.line(im_with_keypoints,(startX,startY),(endX,endY), (0,0 ,255 ), 1)
                #方框
                startX=int(Center[0])-SideLong
                startY=int(Center[1])-SideLong
                endX=int(Center[0])+SideLong
                endY=int(Center[1])+SideLong
                cv2.rectangle(im_with_keypoints, (startX, startY), (endX, endY), (0, 0 ,255), 1)
            
        points2f = cv2.KeyPoint_convert(keypoints)
        drawGG(points2f)
        # Show blobs
        cv2.imshow("Keypoints", im_with_keypoints)

        
    def VideoTracking_Func(self):
        #img = cv2.imread("E:\\sift1.jpg")
        cap = cv2.VideoCapture(r"Q2_Image\opticalFlow.mp4")
        
        
        # Shi-Tomasi角点检测相关参数
        lk_params = dict( winSize  = (21, 21),
                          maxLevel = 2, 
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    
 
        
        # 创建随机颜色
        color = np.random.randint(0, 255, (100, 3))
        
        # 获取视频第一帧
        ret, old_frame = cap.read()
        # 转换为灰度
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # ST检测角点，注意这里参数传递的方法
        ret, frame = cap.read()
        im=frame
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
         
         
        # Filter by Area.
        params.filterByArea = True
        #params.maxArea=200
        #params.minArea = 75
        params.maxArea=200
        params.minArea = 30
         
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.75
         
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
            
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
         
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
        	detector = cv2.SimpleBlobDetector(params)
        else : 
        	detector = cv2.SimpleBlobDetector_create(params)
         
         
        # Detect blobs.
        keypoints = detector.detect(im)
         
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob
         
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        
        #繪製十字及方框
        def drawGG(points2f):
            SideLong=5#邊長
            for Center in points2f:
                print(Center)#座標
                print(Center[0],Center[1])
                #橫
                startX=int(Center[0])-SideLong
                startY=int(Center[1])
                endX=int(Center[0])+SideLong
                endY=int(Center[1])
                #cv2.line(im_with_keypoints,(startX,startY),(endX,endY), (0, 255, 0),1)
                #直
                startX=int(Center[0])
                startY=int(Center[1])-SideLong
                endX=int(Center[0])
                endY=int(Center[1])+SideLong
                #cv2.line(im_with_keypoints,(startX,startY),(endX,endY), (0,255 ,0 ), 1)
                #方框
                startX=int(Center[0])-SideLong
                startY=int(Center[1])-SideLong
                endX=int(Center[0])+SideLong
                endY=int(Center[1])+SideLong
                cv2.rectangle(im_with_keypoints, (startX, startY), (endX, endY), (255, 255 ,255),3 )
            
        points2f = cv2.KeyPoint_convert(keypoints)
        drawGG(points2f)
        # Show blobs
        #cv2.imshow("Keypoints", im_with_keypoints)
        img=im_with_keypoints
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 第一个参数是输入图像
        # 第二个参数是检测的角点数目
        # 第三个参数是角点质量
        # 第四个参数是角点间的最短距离
        p0 = cv2.goodFeaturesToTrack(gray, 7, 0.5, 15)
        #p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        # 创建一个新的图片用于绘制
        mask = np.zeros_like(old_frame)
        
        while 1:
            ret, frame = cap.read()
        
            if frame is None:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break
            else:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 计算光流
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
                # 选择好的特征点
                if p1 is None:
                    pass
                elif p0 is None:
                    pass
                else:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
        
                # 输出每一帧内特征点的坐标
                # 坐标个数为之前指定的个数
                print (good_new)
        
                # 绘制轨迹
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 2, color[i].tolist(), -1)
                img = cv2.add(frame, mask)
        
                cv2.imshow('frame', img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        
                # 更新上一帧以及特征点
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
        
        cv2.destroyAllWindows()
        cap.release()

    def Image_Reconstruction_Func(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import minmax_scale
        images = glob.glob(r"Q4_Image\*.jpg")
        origin_image_array = []
        reconstruction_image_array = []
        for fname in images:
            color_img = cv2.imread(fname)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            width, height, dimension= np.shape(color_img)
            origin_image_array.append(color_img)
            
            def pca_process(input_image):
                image_index, width, height, dimension = np.shape(input_image)
                flatten_image = np.reshape(input_image, (input_image.shape[0], -1))

                pca = PCA(n_components=50,
                        random_state=9527)
                pipe = make_pipeline(StandardScaler(), pca)
                print(pipe)
                transform_image = pca.fit_transform(flatten_image)
                reconstruction_image = pca.inverse_transform(transform_image)
                reconstruction_image = minmax_scale(pca.components_, axis = 1)
                print(reconstruction_image.shape)
                reconstruction_image = np.reshape(reconstruction_image, (width, height, dimension))
                normalizeImage = np.zeros((width, height, dimension))
                reconstruction_image = cv2.normalize(reconstruction_image, normalizeImage, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # pca = PCA(component_rate)
                # lower_dimension_data = pca.fit_transform(input_image)
                # approximation = pca.inverse_transform(lower_dimension_data)
                # approximation = np.reshape(approximation, (width, height))
                # normalizeImage = np.zeros((width, height))
                # approximation = cv2.normalize(approximation, normalizeImage, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                return reconstruction_image

            # R_dimension = color_img[:, :, 0]
            # G_dimension = color_img[:, :, 1]
            # B_dimension = color_img[:, :, 2]

            # approximation_R = pca_process(R_dimension, 0.85)
            # approximation_G = pca_process(G_dimension, 0.85)
            # approximation_B = pca_process(B_dimension, 0.85)

            # reconstructionImage = np.dstack((approximation_R, approximation_G, approximation_B))
            reconstructionImage = pca_process(color_img)
            reconstruction_image_array.append(reconstructionImage)        

        fig, axs= plt.subplots(4, 17, figsize = [50, 15])
        axs[0,0].set_ylabel('Origin')
        axs[1,0].set_ylabel('Recontruction')
        axs[2,0].set_ylabel('Origin')
        axs[3,0].set_ylabel('Recontruction')

        for i in range(17):
            image_A = origin_image_array[i]
            reImage_A = reconstruction_image_array[i]
            axs[0, i].imshow(image_A)
            axs[1, i].imshow(reImage_A)

            image_B = origin_image_array[i+17]
            reImage_B = reconstruction_image_array[i+17]
            axs[2, i].imshow(image_B)
            axs[3, i].imshow(reImage_B)

        plt.show()
    
    def Calculate_Reconstruction_Error_Func(self):
        from numpy.testing import assert_array_almost_equal
        from sklearn.decomposition import PCA
        images = glob.glob(r"Q4_Image\*.jpg")
        origin_image_array = []
        reconstruction_image_array = []
        for fname in images:
            color_img = cv2.imread(fname)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            width, height, dimension= np.shape(color_img)
            
            def pca_process(input_image, component_rate):
                width, height= np.shape(input_image)
                pca = PCA(component_rate)
                lower_dimension_data = pca.fit_transform(input_image)
                approximation = pca.inverse_transform(lower_dimension_data)
                approximation = np.reshape(approximation, (width, height))
                normalizeImage = np.zeros((width, height))
                approximation = cv2.normalize(approximation, normalizeImage, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                return approximation

            R_dimension = color_img[:, :, 0]
            G_dimension = color_img[:, :, 1]
            B_dimension = color_img[:, :, 2]

            approximation_R = pca_process(R_dimension, 0.85)
            approximation_G = pca_process(G_dimension, 0.85)
            approximation_B = pca_process(B_dimension, 0.85)

            reconstructionImage = np.dstack((approximation_R, approximation_G, approximation_B))
            reconstructionImage_flatten = reconstructionImage.flatten()
            color_img_flatten = color_img.flatten()
            error_value = np.linalg.norm(color_img_flatten - reconstructionImage_flatten, axis=0).sum()/30000
            print(fname + ' Reconstruction Error: ', error_value)
                



