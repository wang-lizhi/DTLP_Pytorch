# DTLP
PyTorch codes for paper: **Shipeng Zhang, Lizhi Wang, Lei Zhang, and Hua Huang, Learning Tensor Low-Rank Prior for Hyperspectral Image Reconstruction, CVPR, 2021.**[[Link]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Tensor_Low-Rank_Prior_for_Hyperspectral_Image_Reconstruction_CVPR_2021_paper.pdf)

## Abstract
Snapshot hyperspectral imaging has been developed to capture the spectral information of dynamic scenes. In this paper, we propose a deep neural network by learning the tensor low-rank prior of hyperspectral images (HSI) in the feature domain to promote the reconstruction quality. Our method is inspired by the canonical-polyadic (CP) decomposition theory, where a low-rank tensor can be expressed as a weight summation of several rank-1 component tensors. Specifically, we first learn the tensor low-rank prior of the image features with two steps: (a) we generate rank-1 tensors with discriminative components to collect the contextual information from both spatial and channel dimensions of the image features; (b) we aggregate those rank-1 tensors into a low-rank tensor as a 3D attention map to exploit the global correlation and refine the image features. Then, we integrate the learned tensor low-rank prior into an iterative optimization algorithm to obtain an end-to-end HSI reconstruction. Experiments on both synthetic and real data demonstrate the superiority of our method. 

## Data
In the paper, two benchmarks are utilized for training and testing. [Harvard Dataset](http://vision.seas.harvard.edu/hyperspec/), which is one of them, is used for reproduction. In addition, an extra-experiment following [TSA-Net](https://link.springer.com/chapter/10.1007%2F978-3-030-58592-1_12) is implemented on [CAVE Dataset](https://www1.cs.columbia.edu/CAVE/projects/gap_camera/) and [KAIST Dataset](http://vclab.kaist.ac.kr/siggraphasia2017p1/). To start your work, make HDF5 files of the same length and place them in the correct path. The file structure is as follows:<br/>
>--data/<br/>
>>--Havard_train/<br/>
>>>--trainset_1.h5<br/>
>>>...<br/>
>>>--trainset_n.h5<br/>
>>>--train_files.txt<br/>
>>>--validset_1.h5<br/>
>>>...<br/>
>>>--validset_n.h5<br/>
>>>--valid_files.txt<br/>

>>--Havard_test/<br/>
>>>--test1/<br/>
>>>...<br/>
>>>--testn/<br/>

A few descriptions of datasets can be checked in [README](https://github.com/MaxtBIT/DTLP_PyTorch/blob/main/data/readme.txt). Note that, every image for testing is saved as several 2D images according to different channels.

## Environment
Python 3.6.2<br/>
CUDA 10.0<br/>
Torch 1.7.0<br/>
OpenCV 4.5.4<br/>
h5py 3.1.0<br/>
TensorboardX 2.4<br/>
spectral 0.22.4<br/>

## Usage
1. Download this repository via git or download the [ZIP file](https://github.com/MaxtBIT/DTLP_PyTorch/archive/refs/heads/main.zip) manually.
```
git clone https://github.com/MaxtBIT/DTLP_PyTorch.git
```
2. Download the [pre-trained models](https://drive.google.com/file/d/1Pk0SF6W4pXkLzgU6TCWuXs_5hN387ehM/view?usp=sharing) if you need.
3. Make the datasets and place them in correct paths. Then, adjust the settings in **utils.py** according to your data.
4. Run the file **main.py** to train a model.
5. Run the files **test_for_paper.py** and **test_for_kaist.py** to test models.

## Results
### 1. Reproducing Results on Harvard Dataset
The results reproduced on [Harvard Dataset](http://vision.seas.harvard.edu/hyperspec/). In this stage, the mask is randomly generated for each batch. And the size of patches is 48 * 48 * 31. In addition, only the central areas with 256 * 256 * 31 are compared in testing.
<table align="center">
   <tr align = "center">
      <td></td>
      <td>Paper</td>
      <td>Reproducing</td>
   </tr>
   <tr align = "center">
      <td>PSNR</td>
      <td>32.43</td>
      <td>32.22</td>
   </tr>
   <tr align = "center">
      <td>SSIM</td>
      <td>0.941</td>
      <td>0.936</td>
   </tr>
   <tr align = "center">
      <td>SAM</td>
      <td>0.090</td>
      <td>0.067</td>
   </tr>
</table>

### 2. Results of Extra-Experiments on CAVE&KAIST Datasets
For academic reference, we have added some comparisons with the latest methods on [CAVE Dataset](https://www1.cs.columbia.edu/CAVE/projects/gap_camera/) and [KAIST Dataset](http://vclab.kaist.ac.kr/siggraphasia2017p1/). Methods for comparison include [TSA](https://github.com/mengziyi64/TSA-Net/), [DGSM](https://github.com/TaoHuang95/DGSMP) and [DSSP](https://github.com/wang-lizhi/DSSP), and  our method is completely consistent with the experimental setup of these methods. In addition, we have also increased the comparison of using different masks. In "Real-mask", a given real mask in the range of 0-1 is utilized, which is provided by [TSA](https://github.com/mengziyi64/TSA-Net/tree/master/TSA_Net_realdata/Data). In "Binary-mask", the given real mask is rounded to a binary mask. When training the model, a 48 * 48 sub-mask should be randomly derived from the given real mask for each batch. Note that, images with a size of 256 * 256 * 28, matched the given real mask, are used for comparison.
<table align="center">
   <tr align = "center">
      <td  rowspan="2"></td>
      <td>TSA</td>
      <td>DGSM</td>
      <td colspan="2">DSSP</td>
      <td colspan="2">DTLP</td>
   </tr>
   <tr align = "center">
      <td>Real-mask</td>
      <td>Real-mask</td>
      <td>Real-mask</td>
      <td>Binary-mask</td>
      <td>Real-mask</td>
      <td>Binary-mask</td>
   </tr>
   <tr align = "center">
      <td>PSNR</td>
      <td>31.46</td>
      <td>32.63</td>	
      <td>32.39</td>
      <td>32.84</td>
      <td>33.88</td>
      <td>34.07</td>
   </tr>
   <tr align = "center">
      <td>SSIM</td>
      <td>0.894</td>
      <td>0.917</td>
      <td>0.971</td>
      <td>0.974</td>
      <td>0.926</td>
      <td>0.929</td>
   </tr>
   <tr align = "center">
      <td>SAM</td>
      <td>-</td>
      <td>-</td>
      <td>0.177</td>
      <td>0.163</td>
      <td>0.099</td>
      <td>0.097</td>
   </tr>
</table>

## Citation
```
@inproceedings{DTLP,
  title={Learning Tensor Low-Rank Prior for Hyperspectral Image Reconstruction},
  author={Zhang, Shipeng and Wang, Lizhi and Zhang, Lei and Huang, Hua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12006--12015},
  year={2021}
}
```
