from Detector import *
import os


videoPath = 0 ;
# videoPath = "D:\\projects\\portfolio\\assets\\object\\model_data\\test_videos\\street2.mp4"
    # if you want to
    #   run it on your webcam, change the value to 0
configPath = "D:\\projects\\portfolio\\assets\\object\\model_data\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
modelPath= "D:\\projects\\portfolio\\assets\\object\\model_data\\frozen_inference_graph.pb"
classesPath = "D:\\projects\\portfolio\\assets\\object\\model_data\\coco.names"
def main():
    Detector(videoPath, configPath, modelPath, classesPath)

if __name__ == '__main__':
    main()
    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()
