# %%
## Train the model
# ./darknet detector train data.txt config.txt darknet53.conv.74 

## mAP for the trained model
# ./darknet detector map data.txt config.txt backup/config_last.weights

## Anchor boxes
# ./darknet detector calc_anchors data.txt -num_of_clusters 6 -height 416 -width 416

## Test with webcam
# ./darknet detector demo data.txt config.txt backup/ccpd_1.weights

## Test on a video and save it
# ./darknet detector demo data.txt config.txt backup/config_last.weights test_img_vid/1.mp4 -out_filename 1.mp4
# 
## FPS
# ./darknet detector demo data.txt config.txt backup/config_last.weights 2.mp4 -dont_show -ext_output

# ./darknet detector test data.txt config.txt backup/config_last.weights -dont_show -ext_output < test.txt > result.txt