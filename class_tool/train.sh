python class_tool/train_class.py --data /home2/ImageNet --out-dir class_tool/weights/pva --lr .001 -b 64 -ba 2 --gpu 2 2>&1 | tee pva_2019_07_27

#python main.py --data /home2/project_data/BreakPaper/FDY_POY_cam12_inside/ -f 5 -a vgg16_lpf --out-dir weights/vgg16_lpf5 --lr .001 -b 64 -ba 2 2>&1 | tee paper_vgg16_lpf5_2019_05_30