# Sports Video Analysis on Large-Scale Data

*[Dekun Wu](http://www.cse.yorku.ca/~hadjisma/)*<sup>1*</sup>, 
*[He Zhao](https://joehezhao.github.io/)*<sup>2*</sup>, 
*[Xingce Bao](https://thoth.inrialpes.fr/people/mdvornik/)*<sup>3</sup>, 
*[Richard P. Wildes](http://www.cse.yorku.ca/~wildes/)*<sup>2</sup>, 

<sup>1</sup>University of Pittsburgh &nbsp;&nbsp;
<sup>2</sup>York University &nbsp;&nbsp; 
<sup>3</sup>EPFL &nbsp;&nbsp; 

<span>*</span> Equal contribution

<div align="center">
<img src="img/ECCV2022_TeaserFigure.jpg" width=450px></img>
</div>

**Abstract**: This paper investigates the  modeling of automated machine description on sports video, which has seen much progress recently. Nevertheless, state-of-the-art approaches fall quite short of capturing how human experts analyze sports scenes. In this paper, we propose a novel large-scale NBA dataset for Sports Video Analysis (NSVA) with a focus on captioning, to address the above challenges. We also design a unified approach to process raw videos into a stack of meaningful features with minimum labelling efforts, showing that cross modeling on such features using a transformer architecture leads to strong performance. In addition, we demonstrate the broad application of NSVA by addressing two additional tasks, namely fine-grained sports action recognition and salient player identification.

## Algorithm outline
<div align="center">
<img src="img/ECCV2022_Algorithm.jpg" width=550px></img>
</div>

**Approach**: Our approach relies on feature representations extracted from multiple orthogonal perspectives, we adopt the framework of UniVL [1], a network designed for cross feature interactive modeling, as our base model. It consists of four transformer backbones that are responsible for coarse feature encoding (using TimeSformer [2]), fine-grained feature encoding (e.g., basket, ball, players), cross attention and decoding, respectively. 

## Code Overview
The following sections contain scripts or PyTorch code for:
A. Download pre-processed NSVA dataset.
B. Training/evaluation script: (1) video captioning, (2) action recognition and (3) player identification.
C. Pre-trained weigths.

## Install Dependencys (Same as UniVL)
* python==3.6.9
* torch==1.7.0+cu92
* tqdm
* boto3
* requests
* pandas
* nlg-eval (Install Java 1.8.0 (or higher) firstly)

```
conda create -n py_univl python=3.6.9 tqdm boto3 requests pandas
conda activate py_univl
pip install torch==1.7.1+cu92
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```
This code assumes CUDA support.

## Prepare the Dataset 
Pleaes download features, organized in pickle files, from the following links and put them in the **data** folder.
```
- TimeSformer          feature: [#1589F0](https://...TimeSformer.pickle)
- CourtLineSeg         feature: https://...Courtlineseg.pickle
- Ball, Basket, Player feature: https://...BAS_BALL_PA.json
```
Note that {Ball, Basket, Player} features are merged together via concatenation.

Please download the following csv/json files and put them in the  **csv** folder.
```
- train files   : https://...train.csv
- test  files   : https://...test.csv
- descriptions  : https://...description.json
```

## 
(i) **Set-up Dataset**. We provide two ways to step-up the dataset for CrossTask [1]. You can **either** use pre-extracted features
```
cd datasets/CrossTask_assets
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip
wget https://vision.eecs.yorku.ca/WebShare/CrossTask_s3d.zip
unzip '*.zip'
```
**or** extract features from raw video using the following code (* Both options work, pick one to use) 
```
cd raw_data_process
python download_CrossTask_videos.py
python InstVids2TFRecord_CrossTask.py
bash lmdb_encode_CrossTask.sh 1 1
```
(ii) **Train and Evaluation**. Set the hyper-variable **train** in  CrossTask_main.py (i.e., the one under **if \_\_name\_\_ == \_\_main\_\_**) to either True or False, to choose between training a network or evaluating a pre-trained model. By default, the code will load the used random datasplit (see datasplit.pth in ./checkpoints) as well as our pre-trained weights (included in ./checkpoints folder).
```
# Set 'train' to (True, False) and then
python CrossTask_main.py
```
(iii) **Results** reproduced from pre-trained model (Numbers may **vary** from runs to runs, due to probalistic sampling)
| **Prediction Horizon T = 3**                      | **Success Rate**  | **mean Accuracy** | **mIoU** |
| -----------------------------| ------- | -------- |----------|
| **Viterbi**                  | **23.40**   | **52.71**    | **73.31**    | 
| **Argmax**                   | 22.27   | 52.64    | 73.28    | 

<!-- | **Prediction Horizon T = 3**                      | **Success Rate**  | **mean Accuracy** | **mIoU** | **NLL** | **ModeCoverPrecision** | **ModeCoverRecall** |
| -----------------------------| ------- | -------- |----------|----------|----------|----------|
| **Viterbi**                  | **23.40**   | **52.71**    | **73.31**    |  4.13 | 35.62 | 66.03|
| **Argmax**                   | 22.27   | 52.64    | 73.28    | - | - | - | -->

## COIN
(i) **Set-up Dataset**. Similarly, to use COIN dataset [2] on our approach, we provide pre-extracted features
```
cd datasets/CrossTask_assets
wget https://vision.eecs.yorku.ca/WebShare/COIN_s3d.zip
unzip '*.zip'
```
or we support extracting features from raw video
```
cd raw_data_process
python download_COIN_videos.py
python InstVids2TFRecord_COIN.py
bash lmdb_encode_COIN.sh 1 1
```
(ii) **Train and Evaluation**. The train/evaluation code for COIN is in the same design as before.
```
python COIN_main.py
```
(iii) **Results** reproduced from pre-trained model. Note that figures in below table are slightly higher than those reported in the paper, as this table comes from one random split and paper reports averaged results from five random split.
| **Prediction Horizon T = 3**                      | **Success Rate**  | **mean Accuracy** | **mIoU** |
| -----------------------------| ------- | -------- |----------|
| **Viterbi**                  | 16.61   | 25.76    | 73.48    |  
| **Argmax**                   | 14.05   | 25.82    | 73.14    | 

## NIV
(i) **Set-up Dataset**. For the NIV dataset [3], either use pre-extracted features
```
cd datasets/NIV_assets
wget https://vision.eecs.yorku.ca/WebShare/NIV_s3d.zip
unzip '*.zip'
```
or extract features from raw video by first downloading videos from official project page
```
cd datasets/NIV_assets/videos
wget https://www.di.ens.fr/willow/research/instructionvideos/data_new.tar.gz
tar -xvzf data_new.tar.gz
find ./data_new -type f -name “*.mpg” | xargs -iF mv F .
```
and then jump to raw_data_process and process raw videos
```
cd raw_data_process
python InstVids2TFRecord_NIV.py
bash lmdb_encode_NIV.sh 1 1
```
(ii) **Train and Evaluation**. The train/evaluation code for NIV is in the same design as before.
```
python NIV_main.py
```

(iii) **Results** reproduced from pre-trained model
| **Prediction Horizon T = 3**                      | **Success Rate**  | **mean Accuracy** | **mIoU** |
| -----------------------------| ------- | -------- |----------|
| **Viterbi**                  | 24.02   | 47.18    | 71.15    |  
| **Argmax**                   | 15.32   | 43.84    | 71.05    | 

## Citation

If you find this code useful in your work then please cite

```bibtex
@inproceedings{he2022p3iv,
  title={P3IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision},
  author={He, Zhao and Hadji, Isma and Nikita, Dvornik and Konstantinos, G., Derpanis and Richard, P., Wildes and Allan, D., Jepson},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  month = {June.},
  year={2022}
}
```

## Contact
Please contact He Zhao @ zhufl@eecs.yorku.ca if any issue.

## References
[1] D. Zhukov et al. "Cross-task weakly supervised learning from instructional videos." CVPR'19.

[2] Y. Tang et al. "COIN: A large-scale dataset for comprehensive instructional video analysis." CVPR'19

[3] JB. Alayrac et al. "Unsupervised learning from narrated instruction videos." CVPR'16.
