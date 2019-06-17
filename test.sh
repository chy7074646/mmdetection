##########################test  cocoData type################################
#tools/dist_test.sh configs/faster_rcnn_r50_fpn_1x.py work_dirs/faster_rcnn_r50_fpn_1x/epoch_1.pth 4 --out work_dirs/faster_rcnn_r50_fpn_1x/test_out.pkl #--eval proposal proposal_fast bbox --show


##########################test  VOCdata type################################
#step 1. generate test_out.pkl
#tools/dist_test.sh configs/faster_rcnn_r50_fpn_1x.py work_dirs/faster_rcnn_r50_fpn_1x/epoch_2.pth 4 --out work_dirs/faster_rcnn_r50_fpn_1x/test_out.pkl #--eval proposal proposal_fast bbox --show

#step 2. get evaluate result according to the test_out.pkl
python tools/voc_eval.py work_dirs/faster_rcnn_r50_fpn_1x/test_out.pkl configs/faster_rcnn_r50_fpn_1x.py