[![Python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch_1.12+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

![ChordGNN_logo](images/chordgnn_logo.png)

# Sequential-ChordGNN

This code is based on the ChordGNN model introduced in the paper [Roman Numeral Analysis with Graph Neural Networks: Onset-Wise Predictions from Note-Wise Features](https://arxiv.org/abs/2307.03544), accepted at ISMIR 2023.

This repository is the fork of the ChordGNN [https://github.com/manoskary/chordgnn](https://github.com/manoskary/chordgnn).

## Abstract
The Roman Numeral Analysis is one of the basic and non-trivial problems in the music theory since it is a universal approach for the deep understanding of the musical piece in terms of harmony for any given orchestration. It is crucial for the identification of the tensions and the relations between neighboring chords. This work further develops the idea of 
*ChordGNN*, a graph neural network with multi-task learning for automated Roman Numeral Analysis. The paper introduces a different sequential architecture (*Sequential-ChordGNN*) that can capture the relations between the subtasks at the early stage, but also in the post-processing. The comparative study includes two models: the original parallel model and the sequential model with a human-expert approach, both with and without post-processing. For the reliability of the experiments, statistical tests were performed. The results show that the domain knowledge used in sequence construction is pivotal for achieving better results in automatic Roman Numeral Analysis, as it overcomes the parallel model and the sequential one with the greedy construction algorithm. 

## Installation

Before starting, make sure to have [conda](https://docs.conda.io/en/latest/miniconda.html) installed.

First, create a new environment for Sequential-ChordGNN:

```shell
conda create -n sequential-chordgnn python=3.8
```

Then, activate the environment:

```shell
conda activate sequential-chordgnn
```


If you have a GPU available you might want to install the GPU dependencies follow [this link](https://pytorch.org/) to install the appropriate version of Pytorch:
In general for CPU version:
```shell
conda install pytorch==1.12.0 cpuonly -c pytorch
```

Finally, clone this repository, move to its directory and install the requirements:

```shell
git clone https://github.com/miksik98/sequential-chordgnn
cd sequential-chordgnn
pip install -r requirements.txt
```

## Train ChordGNN

Training a model from scratch generally requires downloading the training data. However, we provide a dataset class that will handle the data downloading and preprocessing.

To train Sequential-ChordGNN from scratch use:

```shell
python ./sequential-chordgnn/train/chord_prediction.py
```

Use -h to see all the available options. Especially look at the `--mlp-tasks-order` that sets the sequential model introduced in this work.


## Analyse Scores

To analyse a score, you need to provide any Score file, such as a MusicXML file:

```shell
python analyse_score.py --score_path <path_to_score>
```

Check all available options by -h. You can set tasks order in the sequence, also for the post-processing.

The produced analysis will be saved in the same directory as the score, with the same name and the suffix `-analysis`.
The new score will be in the MusicXML format, and will contain the predicted Roman Numeral Analysis as harmony annotations of a new track with the harmonic rhythm as notes.

## Pretrained Models

Two pretrained models are attached to this repository in the ```artifacts``` folder. Model with ID ```4op89cyz``` was trained using following tasks ordering: ```[["pcset","hrhythm","bass","tonkey"],["localkey"],["root"],["inversion"],["degree1"],["degree2","quality"]]```. 
Model with ID ```2pesui9a``` was trained with ```post_process.py``` using model ```4op89cyz``` as base model.

## Aknowledgements

This research was carried out with the support of the Laboratory of Bioinformatics and Computational Genomics and the High Performance Computing Center of the Faculty of Mathematics and Information Science, Warsaw University of Technology.


## Authors

- Jan Mycka (Warsaw University of Technology, Warsaw, Poland)
- Mikołaj Sikora (AGH University of Krakow, Kraków, Poland)
- Maciej Smołka (AGH University of Krakow, Kraków, Poland)
- Jacek Mańdziuk (Warsaw University of Technology, Warsaw, Poland and AGH University of Krakow, Kraków, Poland)
