### Architecture overview of DeepFreqNet
The proposed DeepFreqNet architecture, embodies a carefully engineered multi-scale convolutional
framework optimized for hierarchical feature extraction and dense image classification. Beginning with an input of dimension
32, the network applies a cascade of convolutional layers with progressively smaller kernel sizes 5 × 5, 3 × 3, and 1 × 1 to
capture spatial features ranging from broad contextual patterns to fine-grained details. This multi-scale design enhances the
model’s ability to discern diverse image features critical for accurate classification.
Each convolutional layer is immediately followed by batch normalisation, which normalises the activations and thus accelerates
convergence while improving training stability. Max pooling layers interspersed throughout the network downsample the spatial
dimensions, effectively reducing computational load while preserving salient information. The depth of the convolutional filters
increases stepwise from 64 to 256, facilitating progressively more abstract and discriminative feature representations.
After the convolutional stages, feature maps are flattened and passed through fully connected dense layers augmented with
dropout regularization, which serves to prevent overfitting by randomly deactivating neurons during training. This architectural
composition achieves a harmonious balance between depth and computational complexity, enabling DeepFreqNet to efficiently
learn hierarchical features and deliver robust performance in multi-class image classification tasks.

![image](https://github.com/user-attachments/assets/273549d6-3cf4-4da5-8a8b-e7938b13bbc5)
### Algorithm
The DeepFreqNet architecture's iterative design is explained in Algorithm. It starts with input processing in Block 1, which includes initial convolution and downsampling. In later blocks, it moves on to multi-scale feature extraction, depthwise separable convolution, and residual connections.
![image](https://github.com/user-attachments/assets/7b47675d-0a10-4015-8d9e-4270537fe331)

### Result Analysis
Numerous datasets, including BSL, BSL-40, ISL-alpha char, ASL, KU-BdSL, blood cells, MRI tumour classification, DeepHisto, and tumour classification, were used to assess the DeepFreqNet.  Enumerate the learning rates associated with each dataset's model training and validation accuracy.
![image](https://github.com/user-attachments/assets/a83cb337-04de-400c-b072-de6549bdf90b)
## Qualitative Comparison
Comparing Performance Using Different Models in Different Datasets.
![image](https://github.com/user-attachments/assets/baac1f41-f9fc-413b-9a62-ab66f9c4a820)
![WhatsApp Image 2025-05-31 at 09 41 33_9d9dbef7](https://github.com/user-attachments/assets/cabe316c-ee55-4f33-a603-5619409a9fd0)

### Explainable AI (XAI)
In Figure, we applied explainable AI (XAI) techniques, namely GradCAM and GradCAM++.  By providing insights into intricate deep learning models, these methodologies improve the transparency, interpretability, and dependability of the models.
![image](https://github.com/user-attachments/assets/5b924f9c-92b5-40aa-ad9c-0ed8089fbfb1)

### Installation
For python
pip install python=3.11
For tensorflow
pip install tesorflow
Other dependencies
pip install -r requirements.txt


### Dataset links
 1.Bangla Sign Language (BSL)\\
 https://www.kaggle.com/datasets/smnuruzzaman/bangla-sign-language-bsl/data
 2. BdSL40 Dataset
 https://github.com/Patchwork53/BdSL40_Dataset_AI_for_Bangla_2.0_Honorable_Mention
 3. Indian Sign Language (ISL)
 https://data.mendeley.com/datasets/7tsw22y96w/1
 4. KU-BdSL
 https://data.mendeley.com/datasets/scpvm2nbkm/4
 5. Blood Cell Images
 https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data
 6. Brain Tumor MRI Dataset
 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
 7.DeepHisto
 https://zenodo.org/records/7941080
 8. Brain Tumor Classification
 https://figshare.com/articles/dataset/Machine_Vision_Approach_for_Brain_Tumor_Classification_using_Multi_Features_Dataset/19915306
 9. ASL(American Sign Language) Alphabet Dataset
 https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset
