# %%
from PIL import Image
import os
text_file = open('result.txt')
lines = text_file.readlines()[8:]

img_lines = [f for f in lines if 'jpg' in f]
img_lines_indices = [lines.index(f) for f in img_lines]

os.makedirs('outputs', exist_ok = True)

for i in range(len(img_lines_indices)):
    bbox_cnt = 0
    img_index = img_lines_indices[i]
    try:
        next_img_index = img_lines_indices[i+1]
    except:
        next_img_index = len(lines)
    img_line = lines[img_index]
    img_name = img_line.split(':')[0]
    img = Image.open(img_name)
    
    for line in lines[img_index : next_img_index]:
        if 'Box' in line:
            bbox = line.strip().split(':')[-1].strip().split(',')
            for element in bbox:
                edited_element = element.split('=')[-1]
                bbox[bbox.index(element)] = int(edited_element)
            
            cropped = img.crop(bbox)
            new_name = img_name.split('.')
            new_name = new_name[0] + '_' +str(bbox_cnt) + '.' + new_name[1]
            new_name = os.path.join('outputs', os.path.split(new_name)[-1])
            bbox_cnt +=1
            cropped.save(new_name)
            break
    
    #break
print('Done!')
# %%
