# Iranian License Plate Recognition Using a Reliable Deep Learning Approach
The official code for ["Iranian license plate recognition using a reliable deep learning approach"](https://arxiv.org/abs/2305.02292).

<p align="justify">
In this proposed method, license plate recognition is done in two steps. Firstly, the license plates are detected from the input image using the YOLOv4-tiny model, which is based on the Convolutional Neural Network (CNN). Secondly, the characters on the license plates are recognized using the Convolutional Recurrent Neural Network (CRNN) and Connectionist Temporal Classification (CTC). With no need to segment and label the characters separately, one string of numbers and letters is enough for the labels. The successful training of the models involved using 3065 images of license plates and 3364 images of license plate characters as the desired datasets.
</p>

## An Overview
The overall flow chart of the proposed method is:
<p align="center">
<img width="400" alt="image" src="https://github.com/SoheilaHatami/Persian-License-Plate-Recognition/assets/74190994/00ccabb1-5505-457d-bf26-5b230ffe88cf">
</p>

### Dataset and the Proposed Model
1) Download the related datasets by contacting Me via `hatami.soheila95@gmail.com`. 
2) Browse the three folders for license plate detection `LPD`, character recognition `OCR`, and finally the whole process as `TOTAL`.


## Citation
```
@article{Faculty of Mechanical Engineering, Tarbiat Modares University, Tehran, Iran,
  title={Iranian License Plate Recognition Using a Reliable Deep Learning Approach},
  author={Hatami, Soheila and Sadedel, Majid and Jamali, Farideh},
  journal={arXiv preprint arXiv:2305.02292},
  year={2023}
}
```
