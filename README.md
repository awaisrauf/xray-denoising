# Unsupervsied Denoising of X-Ray Images
This is code for the paper acccepted in NIPS18 Workshop on Machine Learning for Health. Further details [link](https://awaisrauf.github.io/xray-denoising).
## Abstract
Among the plethora of techniques devised to curb the prevalence of noise in medical images, deep learning based approaches have shown most promise. However, one critical limitation of these deep learning based denoisers is the requirement of high quality noiseless ground truth images that are difficult to obtain in many medical imaging applications such as X-rays. To circumvent this issue, we leverage recently proposed approach of this [paper](https://arxiv.org/pdf/1803.01314.pdf) that incorporates Stein's Unbiased Risk Estimator (SURE) to train a deep convolutional neural network without requiring denoised ground truth X-ray data.
## Results
Some qualitative results are shown here. For more results, see paper. 
![](imgs/NLM.png)

![](imgs/CHEST.png)

### References 
This code is heavily based on following repository.
https://github.com/ricedsp/D-AMP_Toolbox/tree/master/LDAMP_TensorFlow
