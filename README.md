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
