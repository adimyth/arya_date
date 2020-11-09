# DATE OCR
## Problem Description
### Generalise for unseen positions in date

In any instruments(cheques,forms etc) the date is a mandatory field, and the same needs to be
extracted with maximum accuracy to automate use cases related to it.
In the real-world scenario, we receive data which is biased positionally, for example the ​ **month**
would be only from 01-12, hence we don’t have training data that covers all the possible
combinations across all positions of DDMMYYYY.
But while inference the model should not be biased towards certain digits across positions as It
may bypass many invalid dates
The training and test dataset is specially synthesized to test the generalisation of model across
positions, as some of the positions in training data is heavily biased for certain digit/digits

Link to download required data - ​[http://13.234.225.243:9600](http://13.234.225.243:9600)

The training and test images can be extracted from the respective tar files.

## Preprocessing & Data Augmentation

## OCR
OCR usually has two stages -
* **Text Detection** - Text Detection is minimal here, as the dates are already focused upon. Additional cleaning steps such as removing whitespaces, extracting datebox is applied.
* **Text Recognition** -

### CRNN
There are many approaches discussed in [this](https://towardsdatascience.com/a-gentle-introduction-to-ocr-ee1469a201aa) article. However, the *Specialized DL Approaches* & *Standard DL Approaches* require converting data in one form or the other. CRNN works on the dataset provided out of the box.

CRNN model consisting of a CNN & RNN blocks along with a Transcription Layer.
* CNN Block - Two convolution & maxpool layer. Extracts features from image
* RNN Block - Two Bidirectional LSTM layers. Splits features into some feature sequences & pass it to recurrent layers
* Transcription Layer - Conversion of Feature-specific predictions to Label using CTC. CTC loss is specially designed to optimize both the *length* of the predicted sequence and the *classes* of the predicted sequence


![CRNN](https://miro.medium.com/max/894/0*nGWtig3Cd0Jma2nX)

* Training & Inference Notebook - [digits_ocr_kfold | Kaggle Kernel Noteboook](https://www.kaggle.com/aditya08/digits-ocr/)
* Model weights & submissions - [digits_ocr_kfold | Kaggle Kernel Output](https://www.kaggle.com/aditya08/digits-ocr/output)

`"score": {"acc_8":0.69, "acc_7":0.791, "acc_5":0.92}`

## OCR K-Fold
### Model
The model is the same as above

### Stratification
Trained a model per fold using [MultiLabelStratifiedKFold](https://github.com/trent-b/iterative-stratification). Stratification was done on the basis of digits distribution per example in the training data. I chose number of folds=5.

### Ensembling per fold result
**Majority Voting** - If 3 or more models have the same prediction, then make it the final prediction
**Positionwise Majority Voting** - For other cases (# common preds <=2), choose the digit which occurs the most at a given index. This is the same as above technique but at a more granular level.
Refer [ensemble_folds.ipynb](ensemble_folds.ipynb) for the implementation

* Training & Inference Notebook - [digits_ocr_kfold | Kaggle Kernel Noteboook](https://www.kaggle.com/aditya08/digits-ocr-kfold?scriptVersionId=46308584)
* Model weights & submissions - [digits_ocr_kfold | Kaggle Kernel Output](https://www.kaggle.com/aditya08/digits-ocr-kfold/output?scriptVersionId=46308584)

`"score": {"acc_8":0.698, "acc_7":0.778, "acc_5":0.946}`

## Things to Try
* Use larger model with more number of convolution blocks. Apply BatchNormalization after each convolution block
* For some cases the model predicts sequences of length other than 8. Maybe playing around with the number of units in the last Bi-LSTM layer might help
* CRNN with Spatial Transformer - https://github.com/sbillburg/CRNN-with-STN
* Use model weights trained from CRNN on SVHN dataset - https://github.com/kfengtee/crnn-license-plate-OCR. (Based on torch)
* Some augmentation techniques used by participants in [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)

## Related Work
[SEE](https://arxiv.org/pdf/1712.05404.pdf) — Semi-Supervised End-to-End Scene Text Recognition too works just with text annotation & doesn't require bounding boxes.

## Image Captioning
Treated the problem as an image captioning problem. Code was taken from [tensorflow's image captioning tutorial](https://www.tensorflow.org/tutorials/text/image_captioning). Experimented with EfficientNetB3 along with InceptionV3 as the base model for feature extraction.

This technique completely failed to correctly classify a sequence even on 1% of the testing data. My notebook can be found [here](notebooks/image_captioning.ipynb).

Using EfficientNetB3 gave minimal better positionwise accuracy compared to InceptionV3 model.