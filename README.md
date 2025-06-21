Insect bite classification using deep learning models trained on images across diverse skin tones.  
Thesis project for undergraduate honors research in 2025.  
This repo contains the dataset and models: custom CNN, InceptionV3, and DenseNet169.   
  
Datasets: 
  bite_dataset – Primary dataset of insect bite images (ant, bed bug, chigger, mosquito, and tick bites).  
  bite_dataset_fst – Insect bite dataset that tests the models for Fitzpatrick skin types III - VI to evaluate performance of bites on dark skin tones.  
  ddi_dataset – Stanford DDI (Diverse Dermatology Images) dataset used for pretraining to improve generalization on dark skin tones and conditions. https://ddi-dataset.github.io/index.html#dataset  
  
Within the Models directory:  
  
CNN:   
  CNN.py – Trains a custom Convolutional Neural Network on the insect bite dataset.  
  CNN_fst.py – Tests the CNN model with the bite_dataset_fst to evaluate the model's performance on dark skin tones.  
  CNN_fst_finetune.py – Improves the model by transferring knowledge from a DDI dataset before training on the insect bite dataset.  
  
InceptionV3:   
  InceptionV3.py – Trains an InceptionV3 model on the insect bite dataset.  
  InceptionV3_fst.py - Tests the InceptionV3 model with the bite_dataset_fst to evaluate the model's performance on dark skin tones.  
  InceptionV3_fst_finetune.py – Improves an the model by transferring knowledge from a DDI dataset before training on the insect bite dataset.  
  ta_add.py – Similair to above but adds non-output weights.  
  ta_avg.py – Implements task arithmetic by averaging non-output weights from DDI and insect models to enhance bite classification performance.  
  
DenseNet169:   
  DenseNet.py – Trains the model on the insect bite dataset.  
  DenseNet_finetuned.py – Improves the model by transferring knowledge from a DDI dataset before training on the insect bite dataset.  
  DenseNet_add.py – Implements task arithmetic by adding non-output weights from DDI and insect models to enhance insect bite classification performance.  

  To run the models, please modify the file paths to the datasets in the code to reflect your system.
  
