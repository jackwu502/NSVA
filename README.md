# Sports Video Analysis on Large-Scale Data

*[Dekun Wu](https://jackwu502.github.io/)*<sup>1*</sup>, 
*[He Zhao](https://joehezhao.github.io/)*<sup>2*</sup>, 
*[Xingce Bao](https://www.linkedin.com/in/xingce-bao?originalSubdomain=sg)*<sup>3</sup>, 
*[Richard P. Wildes](http://www.cse.yorku.ca/~wildes/)*<sup>2</sup>, 

<sup>1</sup>University of Pittsburgh &nbsp;&nbsp;
<sup>2</sup>York University &nbsp;&nbsp; 
<sup>3</sup>EPFL &nbsp;&nbsp; 

<span>*</span> Equal contribution
<div align="center">
<img src="img/ECCV2022_TeaserFigure.jpg" width=400px></img>
</div>

**Abstract**: This paper investigates the  modeling of automated machine description on sports video, which has seen much progress recently. Nevertheless, state-of-the-art approaches fall quite short of capturing how human experts analyze sports scenes. In this paper, we propose a novel large-scale NBA dataset for Sports Video Analysis (NSVA) with a focus on captioning, to address the above challenges. We also design a unified approach to process raw videos into a stack of meaningful features with minimum labelling efforts, showing that cross modeling on such features using a transformer architecture leads to strong performance. In addition, we demonstrate the broad application of NSVA by addressing two additional tasks, namely fine-grained sports action recognition and salient player identification.

## Algorithm outline
<div align="center">
<img src="img/ECCV2022_Algorithm.jpg" width=550px></img>
</div>

**Approach**: Our approach relies on feature representations extracted from multiple orthogonal perspectives, we adopt the framework of UniVL [1], a network designed for cross feature interactive modeling, as our base model. It consists of four transformer backbones that are responsible for coarse feature encoding (using TimeSformer [2]), fine-grained feature encoding (e.g., basket, ball, players), cross attention and decoding, respectively. 

## Code Overview
The following sections contain scripts or PyTorch code for:

- A. Download pre-processed NSVA dataset.
- B. Training/evaluation script: (1) video captioning, (2) action recognition and (3) player identification.
- C. Pre-trained weigths.

## Install Dependencys
* python==3.6.9
* torch==1.7.0+cu92
* tqdm
* boto3
* requests
* pandas
* nlg-eval (Install Java 1.8.0 (or higher) firstly)

```
conda create -n sportsformer python=3.6.9 tqdm boto3 requests pandas
conda activate sportsformer
pip install torch==1.7.1+cu92
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```
This code assumes CUDA support.

## Prepare the Dataset 
(1) Pleaes download features, organized in pickle files, from the following links and put them in the **data** folder.
```
- TimeSformer          feature: https://...TimeSformer.pickle
- CourtLineSeg         feature: https://...Courtlineseg.pickle
- Ball, Basket, Player feature: https://...BAS_BALL_PA.json
```
Note that {Ball, Basket, Player} features are merged together via concatenation.

(2) Please download the following csv/json files and put them in the  **csv** folder.
```
- train files   : https://...train.csv
- test  files   : https://...test.csv
- descriptions  : https://...description.json
```

## Video captioning
Run the following code for training/evaluating from scratch video description captioning
```
python main_task_caption.py
```

Or evalute with our pre-trained model at **weights** folder:
```
python main_task_caption.py --eval -pretrained_weight ./weights/ckpt_caption.pkl
```

**Results** reproduced from pre-trained model 

| **Description Captioning**  | **C**  | **M** | **B@1** | **B@2** | **B@3** | **B@4** | **R_L** |
| -----------------------------| ------- | -------- |----------| ----------| ----------| ----------| ----------|
| **Our full model** | **1.1329**   | **0.2420**    | **0.5219**    | **0.4080**    |**0.3120,**    |**0.2425**    |**0.5101** |

## Action recognition
Run the following code for training/evaluating from scratch video description captioning
```
python main_task_action.py
```

Or evalute with our pre-trained model at **weights** folder:
```
python main_task_action.py --eval -pretrained_weight ./weights/ckpt_action.pkl
```

**Results** reproduced from pre-trained model 

| **Action Recognition**  | **SuccessRate**  | **mAcc.** | **mIoU** |
| -----------------------------| ------- | -------- |----------| 
| **Our full model** | **1.1329**   | **0.2420**    | **0.5219**    |

## Player identification
Run the following code for training/evaluating from scratch video description captioning
```
python main_task_identification.py
```

Or evalute with our pre-trained model at **weights** folder:
```
python main_task_identification.py --eval -pretrained_weight ./weights/ckpt_identification.pkl
```

**Results** reproduced from pre-trained model 

| **Play Identification**  | **SuccessRate**  | **mAcc.** | **mIoU** |
| -----------------------------| ------- | -------- |----------|
| **Our full model** | **1.1329**   | **0.2420**    | **0.5219**    | 


## Video downloading tools
If you would like to download the raw mp4 videos, you can use the following code
```
```

## Citation

If you find this code useful in your work then please cite

```bibtex
@inproceedings{dew2022sports,
  title={Sports Video Analysis on Large-Scale Data},
  author={Wu, Dekun and Zhao, He and Bao, Xingce and Wildes, Richard P.},
  booktitle={ECCV},
  month = {Oct.},
  year={2022}
}
```

## Acknowledgement
This code base is largely from [UniVL](https://github.com/microsoft/UniVL). Many thanks to the authors.


## Contact
Please contact Dekun Wu @ dew101@pitts.edu or He Zhao @ zhufl@eecs.yorku.ca, if any issue.

## References
[1] H. Luo et al. "UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation
" Arxiv'2020.

[2] G Bertasiuset al. "Is space-time attention all you need for video understanding?." ICML'2021
