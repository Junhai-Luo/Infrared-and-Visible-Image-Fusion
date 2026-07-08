# Infrared and Visible Image Fusion Papers



## Table of Contents

- [0. Survey / Resource](#0-survey--resource)
- [1. Traditional / Optimization-based Methods](#1-traditional--optimization-based-methods)
- [2. Auto-Encoder / Encoder-Decoder-based Methods](#2-auto-encoder--encoder-decoder-based-methods)
- [3. CNN / Representation Learning-based Methods](#3-cnn--representation-learning-based-methods)
- [4. GAN-based Methods](#4-gan-based-methods)
- [5. Transformer-based Methods](#5-transformer-based-methods)
- [6. Diffusion / Generative / Foundation Model-based Methods](#6-diffusion--generative--foundation-model-based-methods)
- [7. Mamba / State Space Model-based Methods](#7-mamba--state-space-model-based-methods)
- [8. Data Compatibility / Registration / Robustness](#8-data-compatibility--registration--robustness)
- [9. Task-adaptive / Application-oriented Fusion](#9-task-adaptive--application-oriented-fusion)
- [10. Datasets](#10-datasets)
- [11. Evaluation Metrics](#11-evaluation-metrics)
- [12. Suggested Survey Taxonomy](#12-suggested-survey-taxonomy)

---

## 0. Survey / Resource



### IVIF Survey: Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption

Jinyuan Liu, Guanyao Wu, Zhu Liu, Di Wang, Zhiying Jiang, Long Ma, Wei Zhong, Xin Fan, Risheng Liu. IEEE TPAMI / arXiv, 2025.

[![Paper][badge-paper]](https://arxiv.org/abs/2501.10761) [![Zoo][badge-zoo]](https://github.com/RollingPlain/IVIF_ZOO)

### Advances and Challenges in Infrared-Visible Image Fusion: A Comprehensive Review of Techniques and Applications

Rongchao Wang, Zhaofa Zhou, Shuhui Li, et al. Artificial Intelligence Review, 2026.

[![Paper][badge-paper]](https://link.springer.com/article/10.1007/s10462-025-11426-0)

### A Review on Infrared and Visible Image Fusion Algorithms Based on Neural Networks

Kaixuan Yang, Wei Xiang, Zhenshuai Chen, Jian Zhang, Yunpeng Liu. Journal of Visual Communication and Image Representation, 2024.

[![Paper][badge-paper]](https://www.sciencedirect.com/science/article/pii/S1047320324001342)

### Infrared and Visible Image Fusion: A Survey of Current Research Status

C. Lu, et al. ACM, 2024.

[![Paper][badge-paper]](https://dl.acm.org/doi/10.1145/3671151.3671306)

### Infrared and Visible Image Fusion: Methods, Datasets, Applications, and Prospects

Y. Luo, Z. Luo. Applied Sciences, 2023.

[![Paper][badge-paper]](https://www.mdpi.com/2076-3417/13/19/10891)

### Infrared and Visible Image Fusion Technology and Application: A Review

Weihong Ma, Kun Wang, Jiawei Li, Simon X. Yang, Junfei Li, Lepeng Song, et al. Sensors, 2023.

[![Paper][badge-paper]](https://www.mdpi.com/1424-8220/23/2/599)

### Image Fusion Meets Deep Learning: A Survey and Perspective

Hui Zhang, Han Xu, Xin Tian, Junjun Jiang, Jiayi Ma. Information Fusion, 2021.

[![Paper][badge-paper]](https://www.sciencedirect.com/science/article/abs/pii/S1566253521001342)

### Infrared and Visible Image Fusion Techniques Based on Deep Learning: A Review

Changqi Sun, Cong Zhang, Naixue Xiong. Electronics, 2020.

[![Paper][badge-paper]](https://www.mdpi.com/2079-9292/9/12/2162)

### Infrared and Visible Image Fusion Methods and Applications: A Survey

Jiayi Ma, Yong Ma, Chang Li. Information Fusion, 2019.

[![Paper][badge-paper]](https://www.sciencedirect.com/science/article/pii/S1566253517307972)

### A Survey of Infrared and Visual Image Fusion Methods

X. Jin, et al. Infrared Physics & Technology, 2017.

[![Paper][badge-paper]](https://www.sciencedirect.com/science/article/abs/pii/S135044951730052X)

### Infrared-Visible Image Fusion Meets Object Detection: Towards Unified Optimization for Multimodal Perception

Xiantai Xiang, Guangyao Zhou, Ben Niu, et al. Remote Sensing, 2025.

[![Paper][badge-paper]](https://www.mdpi.com/2072-4292/17/21/3637)

### VIF-Benchmark

All deep learning-based infrared and visible image fusion algorithms in a unified benchmark framework.

[![GitHub][badge-github]](https://github.com/Linfeng-Tang/VIF-Benchmark)

### IVIF Zoo

A curated repository for IVIF datasets, methods, evaluation metrics, fusion results, detection/segmentation results, and computational efficiency resources.

[![GitHub][badge-github]](https://github.com/RollingPlain/IVIF_ZOO)

### IVIF-Code-Interpretation

Open-source IVIF paper/code collection.

[![GitHub][badge-github]](https://github.com/liushh39/IVIF-Code-Interpretation)

---

## 1. Traditional / Optimization-based Methods

### GTF: Infrared and Visible Image Fusion via Gradient Transfer and Total Variation Minimization

Jiayi Ma, Chen Chen, Chang Li, Jun Huang. Information Fusion, 2016.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Infrared+and+visible+image+fusion+via+gradient+transfer+and+total+variation+minimization) [![Code][badge-code]](https://github.com/jiayi-ma/GTF) [![MATLAB][badge-matlab]](https://github.com/jiayi-ma/GTF)

### LatLRR: Infrared and Visible Image Fusion via Latent Low-Rank Representation

Jiayi Ma, et al. Information Fusion, 2017.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Infrared+and+visible+image+fusion+via+latent+low-rank+representation) ![No Code][badge-no-code]

### Hybrid MSD / Sparse / Saliency-guided Fusion

Traditional methods include multi-scale decomposition, wavelet / contourlet / NSCT, sparse representation, low-rank representation, saliency maps, guided filtering, and total variation models.

[![More][badge-more]](https://scholar.google.com/scholar?q=infrared+visible+image+fusion+wavelet+NSCT+sparse+representation+saliency+guided+filtering+survey)

---

## 2. Auto-Encoder / Encoder-Decoder-based Methods

### DLF: Infrared and Visible Image Fusion using a Deep Learning Framework

Hui Li, Xiao-Jun Wu, Josef Kittler. arXiv, 2018.

[![Paper][badge-paper]](https://arxiv.org/abs/1804.06992) [![Code][badge-code]](https://github.com/hli1221/imagefusion_deeplearning) [![Python][badge-python]](https://github.com/hli1221/imagefusion_deeplearning)

### DenseFuse: DenseFuse: A Fusion Approach to Infrared and Visible Images

Hui Li, Xiao-Jun Wu. IEEE TIP, 2019.

[![Paper][badge-paper]](https://arxiv.org/abs/1804.08361) [![Code][badge-code]](https://github.com/hli1221/imagefusion_densefuse) [![Python][badge-python]](https://github.com/hli1221/imagefusion_densefuse)

### NestFuse: NestFuse: An Infrared and Visible Image Fusion Architecture based on Nest Connection and Spatial/Channel Attention Models

Hui Li, Xiao-Jun Wu, Josef Kittler. IEEE TIM, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=NestFuse%3A+An+Infrared+and+Visible+Image+Fusion+Architecture+based+on+Nest+Connection+and+Spatial%2FChannel+Attention+Models) [![Code][badge-code]](https://github.com/hli1221/imagefusion-nestfuse) [![Python][badge-python]](https://github.com/hli1221/imagefusion-nestfuse)

### DIDFuse: DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion

Zixiang Zhao, et al. IJCAI, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=DIDFuse%3A+Deep+Image+Decomposition+for+Infrared+and+Visible+Image+Fusion) [![Code][badge-code]](https://github.com/Zhaozixiang1228/IVIF-DIDFuse) [![Python][badge-python]](https://github.com/Zhaozixiang1228/IVIF-DIDFuse)

### SEDRFuse: SEDRFuse: A Symmetric Encoder-Decoder with Residual Block Network for Infrared and Visible Image Fusion

IEEE TIM, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=SEDRFuse%3A+A+Symmetric+Encoder-Decoder+with+Residual+Block+Network+for+Infrared+and+Visible+Image+Fusion) ![No Code][badge-no-code]

### RFN-Nest: RFN-Nest: An End-to-End Residual Fusion Network for Infrared and Visible Images

Hui Li, Xiao-Jun Wu, Josef Kittler. Information Fusion, 2021.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=RFN-Nest%3A+An+End-to-End+Residual+Fusion+Network+for+Infrared+and+Visible+Images) [![Code][badge-code]](https://github.com/hli1221/imagefusion-rfn-nest) [![Python][badge-python]](https://github.com/hli1221/imagefusion-rfn-nest)

### AUIF: Efficient and Interpretable Infrared and Visible Image Fusion via Algorithm Unrolling

IEEE TCSVT, 2021.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Efficient+and+Interpretable+Infrared+and+Visible+Image+Fusion+via+Algorithm+Unrolling) [![Code][badge-code]](https://github.com/Zhaozixiang1228/IVIF-AUIF-Net) [![Python][badge-python]](https://github.com/Zhaozixiang1228/IVIF-AUIF-Net)

### SFAFuse: Self-supervised Feature Adaptation for Infrared and Visible Image Fusion

Information Fusion, 2021.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Self-supervised+Feature+Adaptation+for+Infrared+and+Visible+Image+Fusion) ![No Code][badge-no-code]

### SMoA: SMoA: Searching a Modality-Oriented Architecture for Infrared and Visible Image Fusion

IEEE SPL, 2021.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=SMoA%3A+Searching+a+Modality-Oriented+Architecture+for+Infrared+and+Visible+Image+Fusion) ![No Code][badge-no-code]

### CUFD: CUFD: An Encoder-Decoder Network for Visible and Infrared Image Fusion Based on Common and Unique Feature Decomposition

CVIU, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=CUFD%3A+An+Encoder-Decoder+Network+for+Visible+and+Infrared+Image+Fusion+Based+on+Common+and+Unique+Feature+Decomposition) ![No Code][badge-no-code]

### LRRNet: LRRNet: A Novel Representation Learning Guided Fusion Framework for Infrared and Visible Images

IEEE TPAMI, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=LRRNet%3A+A+Novel+Representation+Learning+Guided+Fusion+Framework+for+Infrared+and+Visible+Images) [![Code][badge-code]](https://github.com/hli1221/imagefusion-LRRNet) [![Python][badge-python]](https://github.com/hli1221/imagefusion-LRRNet)

---

## 3. CNN / Representation Learning-based Methods

### IFCNN: IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network

Information Fusion, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=IFCNN%3A+A+General+Image+Fusion+Framework+Based+on+Convolutional+Neural+Network) [![Code][badge-code]](https://github.com/uzeful/IFCNN) [![Python][badge-python]](https://github.com/uzeful/IFCNN)

### FusionDN: FusionDN: A Unified Densely Connected Network for Image Fusion

AAAI, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=FusionDN%3A+A+Unified+Densely+Connected+Network+for+Image+Fusion) [![Code][badge-code]](https://github.com/hanna-xu/FusionDN) [![Python][badge-python]](https://github.com/hanna-xu/FusionDN)

### PMGI: Rethinking the Image Fusion: A Fast Unified Image Fusion Network Based on Proportional Maintenance of Gradient and Intensity

AAAI, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Rethinking+the+Image+Fusion%3A+A+Fast+Unified+Image+Fusion+Network+Based+on+Proportional+Maintenance+of+Gradient+and+Intensity) [![Code][badge-code]](https://github.com/HaoZhang1018/PMGI_AAAI2020) [![Python][badge-python]](https://github.com/HaoZhang1018/PMGI_AAAI2020)

### U2Fusion: U2Fusion: A Unified Unsupervised Image Fusion Network

IEEE TPAMI, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=U2Fusion%3A+A+Unified+Unsupervised+Image+Fusion+Network) [![Code][badge-code]](https://github.com/hanna-xu/U2Fusion) [![Python][badge-python]](https://github.com/hanna-xu/U2Fusion)

### SDNet: SDNet: A Versatile Squeeze-and-Decomposition Network for Real-Time Image Fusion

IJCV, 2021.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=SDNet%3A+A+Versatile+Squeeze-and-Decomposition+Network+for+Real-Time+Image+Fusion) [![Code][badge-code]](https://github.com/HaoZhang1018/SDNet) [![Python][badge-python]](https://github.com/HaoZhang1018/SDNet)

### STDFusionNet: STDFusionNet: An Infrared and Visible Image Fusion Network Based on Salient Target Detection

IEEE TIM, 2021.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=STDFusionNet%3A+An+Infrared+and+Visible+Image+Fusion+Network+Based+on+Salient+Target+Detection) [![Code][badge-code]](https://github.com/jiayi-ma/STDFusionNet) [![Python][badge-python]](https://github.com/jiayi-ma/STDFusionNet)

### PIAFusion: PIAFusion: A Progressive Infrared and Visible Image Fusion Network Based on Illumination Aware

Linfeng Tang, et al. Information Fusion, 2022.

[![Paper][badge-paper]](https://www.sciencedirect.com/science/article/abs/pii/S156625352200032X) [![Code][badge-code]](https://github.com/Linfeng-Tang/PIAFusion) [![Python][badge-python]](https://github.com/Linfeng-Tang/PIAFusion)

### L2Net: L2Net: Infrared and Visible Image Fusion Using Lightweight Large Kernel Convolution Network

IEEE TIP, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=L2Net%3A+Infrared+and+Visible+Image+Fusion+Using+Lightweight+Large+Kernel+Convolution+Network) ![No Code][badge-no-code]

### IGNet: Learning a Graph Neural Network with Cross Modality Interaction for Image Fusion

ACM MM, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Learning+a+Graph+Neural+Network+with+Cross+Modality+Interaction+for+Image+Fusion) ![No Code][badge-no-code]

### PSFusion: Rethinking the Necessity of Image Fusion in High-Level Vision Tasks: A Practical Infrared and Visible Image Fusion Network Based on Progressive Semantic Injection and Scene Fidelity

Information Fusion, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Rethinking+the+Necessity+of+Image+Fusion+in+High-Level+Vision+Tasks%3A+A+Practical+Infrared+and+Visible+Image+Fusion+Network+Based+on+Progressive+Semantic+Injection+and+Scene+Fidelity) [![Code][badge-code]](https://github.com/Linfeng-Tang/PSFusion) [![Python][badge-python]](https://github.com/Linfeng-Tang/PSFusion)

### LUT-Fuse: LUT-Fuse: Towards Extremely Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up Tables

ICCV, 2025.

[![Paper][badge-paper]](https://arxiv.org/abs/2509.00346) [![Code][badge-code]](https://github.com/zyb5/LUT-Fuse) [![Python][badge-python]](https://github.com/zyb5/LUT-Fuse)

---

## 4. GAN-based Methods

### FusionGAN: FusionGAN: A Generative Adversarial Network for Infrared and Visible Image Fusion

Jiayi Ma, et al. Information Fusion, 2019.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=FusionGAN%3A+A+Generative+Adversarial+Network+for+Infrared+and+Visible+Image+Fusion) [![Code][badge-code]](https://github.com/jiayi-ma/FusionGAN) [![Python][badge-python]](https://github.com/jiayi-ma/FusionGAN)

### DDcGAN: Infrared and Visible Image Fusion via Dual-Discriminator Conditional Generative Adversarial Network

IJCAI, 2019.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Infrared+and+Visible+Image+Fusion+via+Dual-Discriminator+Conditional+Generative+Adversarial+Network) [![Code][badge-code]](https://github.com/hanna-xu/DDcGAN) [![Python][badge-python]](https://github.com/hanna-xu/DDcGAN)

### AttentionFGAN / AtFGAN: AttentionFGAN: Infrared and Visible Image Fusion Using Attention-Based Generative Adversarial Networks

IEEE TMM, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=AttentionFGAN%3A+Infrared+and+Visible+Image+Fusion+Using+Attention-Based+Generative+Adversarial+Networks) ![No Code][badge-no-code]

### DPAL: Infrared and Visible Image Fusion via Detail Preserving Adversarial Learning

Information Fusion, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Infrared+and+Visible+Image+Fusion+via+Detail+Preserving+Adversarial+Learning) ![No Code][badge-no-code]

### D2WGAN: Infrared and Visible Image Fusion Using Dual Discriminators Generative Adversarial Networks with Wasserstein Distance

Information Sciences, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Infrared+and+Visible+Image+Fusion+Using+Dual+Discriminators+Generative+Adversarial+Networks+with+Wasserstein+Distance) ![No Code][badge-no-code]

### GANMcC: GANMcC: A Generative Adversarial Network with Multi-classification Constraints for Infrared and Visible Image Fusion

IEEE TIM, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=GANMcC%3A+A+Generative+Adversarial+Network+with+Multi-classification+Constraints+for+Infrared+and+Visible+Image+Fusion) [![Code][badge-code]](https://github.com/HaoZhang1018/GANMcC) [![Python][badge-python]](https://github.com/HaoZhang1018/GANMcC)

### ICAFusion: Infrared and Visible Image Fusion via Interactive Compensatory Attention Adversarial Learning

IEEE TMM, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Infrared+and+Visible+Image+Fusion+via+Interactive+Compensatory+Attention+Adversarial+Learning) [![Code][badge-code]](https://github.com/Zhishe-Wang/ICAFusion) [![Python][badge-python]](https://github.com/Zhishe-Wang/ICAFusion)

### TCGAN: Transformer Based Conditional GAN for Multimodal Image Fusion

IEEE TMM, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Transformer+Based+Conditional+GAN+for+Multimodal+Image+Fusion) ![No Code][badge-no-code]

### FreqGAN: FreqGAN: Infrared and Visible Image Fusion via Unified Frequency Adversarial Learning

IEEE TCSVT, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=FreqGAN%3A+Infrared+and+Visible+Image+Fusion+via+Unified+Frequency+Adversarial+Learning) [![Code][badge-code]](https://github.com/Zhishe-Wang/FreqGAN) [![Python][badge-python]](https://github.com/Zhishe-Wang/FreqGAN)

### DDBF: Dispel Darkness for Better Fusion: A Controllable Visual Enhancer Based on Cross-modal Conditional Adversarial Learning

CVPR, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Dispel+Darkness+for+Better+Fusion%3A+A+Controllable+Visual+Enhancer+Based+on+Cross-modal+Conditional+Adversarial+Learning) [![Code][badge-code]](https://github.com/HaoZhang1018/DDBF) [![Python][badge-python]](https://github.com/HaoZhang1018/DDBF)

### CCF: Conditional Controllable Image Fusion

NeurIPS, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Conditional+Controllable+Image+Fusion) [![Code][badge-code]](https://github.com/jehovahxu/CCF) [![Python][badge-python]](https://github.com/jehovahxu/CCF)

---

## 5. Transformer-based Methods

### SwinFusion: SwinFusion: Cross-domain Long-range Learning for General Image Fusion via Swin Transformer

IEEE/CAA Journal of Automatica Sinica, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=SwinFusion%3A+Cross-domain+Long-range+Learning+for+General+Image+Fusion+via+Swin+Transformer) [![Code][badge-code]](https://github.com/Linfeng-Tang/SwinFusion) [![Python][badge-python]](https://github.com/Linfeng-Tang/SwinFusion)

### YDTR: YDTR: Infrared and Visible Image Fusion via Y-shape Dynamic Transformer

IEEE TMM, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=YDTR%3A+Infrared+and+Visible+Image+Fusion+via+Y-shape+Dynamic+Transformer) [![Code][badge-code]](https://github.com/tthinking/YDTR) [![Python][badge-python]](https://github.com/tthinking/YDTR)

### IFT: Image Fusion Transformer

ICIP, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Image+Fusion+Transformer) [![Code][badge-code]](https://github.com/Vibashan/Image-Fusion-Transformer) [![Python][badge-python]](https://github.com/Vibashan/Image-Fusion-Transformer)

### CDDFuse: CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion

CVPR, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=CDDFuse%3A+Correlation-Driven+Dual-Branch+Feature+Decomposition+for+Multi-Modality+Image+Fusion) [![Code][badge-code]](https://github.com/Zhaozixiang1228/MMIF-CDDFuse) [![Python][badge-python]](https://github.com/Zhaozixiang1228/MMIF-CDDFuse)

### TGFuse: TGFuse: An Infrared and Visible Image Fusion Approach Based on Transformer and Generative Adversarial Network

IEEE TIP, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=TGFuse%3A+An+Infrared+and+Visible+Image+Fusion+Approach+Based+on+Transformer+and+Generative+Adversarial+Network) ![No Code][badge-no-code]

### CMTFusion: Cross-Modal Transformers for Infrared and Visible Image Fusion

IEEE TCSVT, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Cross-Modal+Transformers+for+Infrared+and+Visible+Image+Fusion) [![Code][badge-code]](https://github.com/seonghyun0108/CMTFusion) [![Python][badge-python]](https://github.com/seonghyun0108/CMTFusion)

### DATFuse: Infrared and Visible Image Fusion via Dual Attention Transformer

IEEE TCSVT, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Infrared+and+Visible+Image+Fusion+via+Dual+Attention+Transformer) ![No Code][badge-no-code]

### Text-IF: Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion

CVPR, 2024.

[![Paper][badge-paper]](https://arxiv.org/abs/2403.16387) [![Code][badge-code]](https://github.com/XunpengYi/Text-IF) [![Python][badge-python]](https://github.com/XunpengYi/Text-IF)

### PromptFusion: PromptFusion: Harmonized Semantic Prompt Learning for Infrared and Visible Image Fusion

IEEE/CAA JAS, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=PromptFusion%3A+Harmonized+Semantic+Prompt+Learning+for+Infrared+and+Visible+Image+Fusion) ![No Code][badge-no-code]

### MaeFuse: MaeFuse: Transferring Omni Features with Pretrained Masked Autoencoders for Infrared and Visible Image Fusion via Guided Training

IEEE TIP, 2025.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=MaeFuse%3A+Transferring+Omni+Features+with+Pretrained+Masked+Autoencoders+for+Infrared+and+Visible+Image+Fusion+via+Guided+Training) [![Code][badge-code]](https://github.com/Henry-Lee-real/MaeFuse) [![Python][badge-python]](https://github.com/Henry-Lee-real/MaeFuse)

---

## 6. Diffusion / Generative / Foundation Model-based Methods

### Dif-Fusion: Dif-Fusion: Towards High Color Fidelity in Infrared and Visible Image Fusion with Diffusion Models

Jun Yue, et al. IEEE TIP, 2023.

[![Paper][badge-paper]](https://arxiv.org/abs/2301.08072) [![Code][badge-code]](https://github.com/GeoVectorMatrix/Dif-Fusion) [![Python][badge-python]](https://github.com/GeoVectorMatrix/Dif-Fusion)

### DDFM: DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion

ICCV, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=DDFM%3A+Denoising+Diffusion+Model+for+Multi-Modality+Image+Fusion) [![Code][badge-code]](https://github.com/Zhaozixiang1228/MMIF-DDFM) [![Python][badge-python]](https://github.com/Zhaozixiang1228/MMIF-DDFM)

### VDMUFusion: VDMUFusion: A Versatile Diffusion Model-Based Unsupervised Framework for Image Fusion

IEEE TIP, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=VDMUFusion%3A+A+Versatile+Diffusion+Model-Based+Unsupervised+Framework+for+Image+Fusion) [![Code][badge-code]](https://github.com/yuliu316316/VDMUFusion) [![Python][badge-python]](https://github.com/yuliu316316/VDMUFusion)

### EMMA: Equivariant Multi-Modality Image Fusion

CVPR, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Equivariant+Multi-Modality+Image+Fusion) [![Code][badge-code]](https://github.com/Zhaozixiang1228/MMIF-EMMA) [![Python][badge-python]](https://github.com/Zhaozixiang1228/MMIF-EMMA)

### FILM: Image Fusion via Vision-Language Model

ICML, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Image+Fusion+via+Vision-Language+Model) [![Code][badge-code]](https://github.com/Zhaozixiang1228/IF-FILM) [![Python][badge-python]](https://github.com/Zhaozixiang1228/IF-FILM)

### TC-MoA: Task-Customized Mixture of Adapters for General Image Fusion

CVPR, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Task-Customized+Mixture+of+Adapters+for+General+Image+Fusion) [![Code][badge-code]](https://github.com/YangSun22/TC-MoA) [![Python][badge-python]](https://github.com/YangSun22/TC-MoA)

### SHIP: Probing Synergistic High-Order Interaction in Infrared and Visible Image Fusion

CVPR, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Probing+Synergistic+High-Order+Interaction+in+Infrared+and+Visible+Image+Fusion) [![Code][badge-code]](https://github.com/zheng980629/SHIP) [![Python][badge-python]](https://github.com/zheng980629/SHIP)

### GIFNet: One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion

CVPR, 2025.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=One+Model+for+ALL%3A+Low-Level+Task+Interaction+Is+a+Key+to+Task-Agnostic+Image+Fusion) [![Code][badge-code]](https://github.com/AWCXV/GIFNet) [![Python][badge-python]](https://github.com/AWCXV/GIFNet)

### SGDFuse: SGDFuse: SAM-Guided Diffusion for High-Fidelity Infrared and Visible Image Fusion

arXiv, 2025.

[![Paper][badge-paper]](https://arxiv.org/abs/2508.05264) [![Code][badge-code]](https://github.com/boshizhang123/SGDFuse) [![Python][badge-python]](https://github.com/boshizhang123/SGDFuse)

---

## 7. Mamba / State Space Model-based Methods

### MambaDFuse: MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion

arXiv, 2024.

[![Paper][badge-paper]](https://arxiv.org/abs/2404.08406) [![Code][badge-code]](https://github.com/Lizhe1228/MambaDFuse) [![Python][badge-python]](https://github.com/Lizhe1228/MambaDFuse)

### S4Fusion: S4Fusion: Saliency-aware Selective State Space Model for Infrared Visible Image Fusion

IEEE TIP, 2025 / arXiv, 2024.

[![Paper][badge-paper]](https://arxiv.org/abs/2405.20881) [![Code][badge-code]](https://github.com/zipper112/S4Fusion) [![Python][badge-python]](https://github.com/zipper112/S4Fusion)

### PMKFuse: Infrared and Visible Image Fusion via the Parallel Mamba-KAN Framework

JOLT, 2025.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Infrared+and+Visible+Image+Fusion+via+the+Parallel+Mamba-KAN+Framework) [![Code][badge-code]](https://github.com/sunyichen1994/PMKFuse) [![Python][badge-python]](https://github.com/sunyichen1994/PMKFuse)

### SMC: Self-supervised Multiplex Consensus Mamba for General Image Fusion

AAAI, 2026.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Self-supervised+Multiplex+Consensus+Mamba+for+General+Image+Fusion) ![No Code][badge-no-code]

---

## 8. Data Compatibility / Registration / Robustness

### UMIR: Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation

CVPR, 2020.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Unsupervised+Multi-Modal+Image+Registration+via+Geometry+Preserving+Image-to-Image+Translation) [![Code][badge-code]](https://github.com/moabarar/nemar) [![Python][badge-python]](https://github.com/moabarar/nemar)

### ReCoNet: ReCoNet: Recurrent Correction Network for Fast and Efficient Multi-Modality Image Fusion

ECCV, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=ReCoNet%3A+Recurrent+Correction+Network+for+Fast+and+Efficient+Multi-Modality+Image+Fusion) [![Code][badge-code]](https://github.com/dlut-dimt/ReCoNet) [![Python][badge-python]](https://github.com/dlut-dimt/ReCoNet)

### SuperFusion: SuperFusion: A Versatile Image Registration and Fusion Network with Semantic Awareness

IEEE/CAA JAS, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=SuperFusion%3A+A+Versatile+Image+Registration+and+Fusion+Network+with+Semantic+Awareness) [![Code][badge-code]](https://github.com/Linfeng-Tang/SuperFusion) [![Python][badge-python]](https://github.com/Linfeng-Tang/SuperFusion)

### UMFusion: Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration

IJCAI, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Unsupervised+Misaligned+Infrared+and+Visible+Image+Fusion+via+Cross-Modality+Image+Generation+and+Registration) [![Code][badge-code]](https://github.com/wdhudiekou/UMF-CMGR) [![Python][badge-python]](https://github.com/wdhudiekou/UMF-CMGR)

### MURF: MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion

IEEE TPAMI, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=MURF%3A+Mutually+Reinforcing+Multi-Modal+Image+Registration+and+Fusion) [![Code][badge-code]](https://github.com/hanna-xu/MURF) [![Python][badge-python]](https://github.com/hanna-xu/MURF)

### SemLA: Semantics Lead All: Towards Unified Image Registration and Fusion from a Semantic Perspective

Information Fusion, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Semantics+Lead+All%3A+Towards+Unified+Image+Registration+and+Fusion+from+a+Semantic+Perspective) [![Code][badge-code]](https://github.com/xiehousheng/SemLA) [![Python][badge-python]](https://github.com/xiehousheng/SemLA)

### IVF-WoReg: A Deep Learning Framework for Infrared and Visible Image Fusion Without Strict Registration

IJCV, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=A+Deep+Learning+Framework+for+Infrared+and+Visible+Image+Fusion+Without+Strict+Registration) ![No Code][badge-no-code]

### PAIFusion / PAIF: PAIF: Perception-Aware Infrared-Visible Image Fusion for Attack-Tolerant Semantic Segmentation

ACM MM, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=PAIF%3A+Perception-Aware+Infrared-Visible+Image+Fusion+for+Attack-Tolerant+Semantic+Segmentation) [![Code][badge-code]](https://github.com/LiuZhu-CV/PAIF) [![Python][badge-python]](https://github.com/LiuZhu-CV/PAIF)

---

## 9. Task-adaptive / Application-oriented Fusion

### 9.1 Object Detection-oriented Fusion

### DetFusion: DetFusion: A Detection-driven Infrared and Visible Image Fusion Network

ACM MM, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=DetFusion%3A+A+Detection-driven+Infrared+and+Visible+Image+Fusion+Network) [![Code][badge-code]](https://github.com/SunYM2020/DetFusion) [![Python][badge-python]](https://github.com/SunYM2020/DetFusion)

### TarDAL: Target-aware Dual Adversarial Learning and a Multi-Scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection

Jinyuan Liu, et al. CVPR, 2022.

[![Paper][badge-paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.html) [![Code][badge-code]](https://github.com/dlut-dimt/TarDAL) [![Python][badge-python]](https://github.com/dlut-dimt/TarDAL)

### MetaFusion: MetaFusion: Infrared and Visible Image Fusion via Meta-Feature Embedding from Object Detection

CVPR, 2023.

[![Paper][badge-paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_MetaFusion_Infrared_and_Visible_Image_Fusion_via_Meta-Feature_Embedding_From_CVPR_2023_paper.html) [![Code][badge-code]](https://github.com/wdzhao123/MetaFusion) [![Python][badge-python]](https://github.com/wdzhao123/MetaFusion)

### MoE-Fusion: Multi-modal Gated Mixture of Local-to-Global Experts for Dynamic Image Fusion

ICCV, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Multi-modal+Gated+Mixture+of+Local-to-Global+Experts+for+Dynamic+Image+Fusion) [![Code][badge-code]](https://github.com/SunYM2020/MoE-Fusion) [![Python][badge-python]](https://github.com/SunYM2020/MoE-Fusion)

### DCEvo: DCEvo: Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion

CVPR, 2025.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=DCEvo%3A+Discriminative+Cross-Dimensional+Evolutionary+Learning+for+Infrared+and+Visible+Image+Fusion) [![Code][badge-code]](https://github.com/Beate-Suy-Zhang/DCEvo) [![Python][badge-python]](https://github.com/Beate-Suy-Zhang/DCEvo)

### TDFusion: Task-driven Image Fusion with Learnable Fusion Loss

CVPR, 2025.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Task-driven+Image+Fusion+with+Learnable+Fusion+Loss) [![Code][badge-code]](https://github.com/HaowenBai/TDFusion) [![Python][badge-python]](https://github.com/HaowenBai/TDFusion)

### 9.2 Semantic Segmentation-oriented Fusion

### SeAFusion: Image Fusion in the Loop of High-Level Vision Tasks: A Semantic-Aware Real-Time Infrared and Visible Image Fusion Network

Information Fusion, 2022.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Image+Fusion+in+the+Loop+of+High-Level+Vision+Tasks%3A+A+Semantic-Aware+Real-Time+Infrared+and+Visible+Image+Fusion+Network) [![Code][badge-code]](https://github.com/Linfeng-Tang/SeAFusion) [![Python][badge-python]](https://github.com/Linfeng-Tang/SeAFusion)

### SegMiF: Multi-interactive Feature Learning and a Full-time Multi-Modality Benchmark for Image Fusion and Segmentation

ICCV, 2023.

[![Paper][badge-paper]](https://arxiv.org/abs/2308.02097) [![Code][badge-code]](https://github.com/JinyuanLiu-CV/SegMiF) [![Python][badge-python]](https://github.com/JinyuanLiu-CV/SegMiF)

### MRFS: MRFS: Mutually Reinforcing Image Fusion and Segmentation

CVPR, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=MRFS%3A+Mutually+Reinforcing+Image+Fusion+and+Segmentation) [![Code][badge-code]](https://github.com/HaoZhang1018/MRFS) [![Python][badge-python]](https://github.com/HaoZhang1018/MRFS)

### SAGE: Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond

CVPR, 2025.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Every+SAM+Drop+Counts%3A+Embracing+Semantic+Priors+for+Multi-Modality+Image+Fusion+and+Beyond) [![Code][badge-code]](https://github.com/RollingPlain/SAGE_IVIF) [![Python][badge-python]](https://github.com/RollingPlain/SAGE_IVIF)

### SDSFusion: SDSFusion: A Semantic-Aware Infrared and Visible Image Fusion Network for Degraded Scenes

IEEE TIP, 2025.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=SDSFusion%3A+A+Semantic-Aware+Infrared+and+Visible+Image+Fusion+Network+for+Degraded+Scenes) ![No Code][badge-no-code]

### 9.3 Salient Object / Multi-task Fusion

### BDLFusion: Bi-level Dynamic Learning for Jointly Multi-Modality Image Fusion and Beyond

IJCAI, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Bi-level+Dynamic+Learning+for+Jointly+Multi-Modality+Image+Fusion+and+Beyond) [![Code][badge-code]](https://github.com/LiuZhu-CV/BDLFusion) [![Python][badge-python]](https://github.com/LiuZhu-CV/BDLFusion)

### IRFS: An Interactively Reinforced Paradigm for Joint Infrared-Visible Image Fusion and Saliency Object Detection

Information Fusion, 2023.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=An+Interactively+Reinforced+Paradigm+for+Joint+Infrared-Visible+Image+Fusion+and+Saliency+Object+Detection) [![Code][badge-code]](https://github.com/wdhudiekou/IRFS) [![Python][badge-python]](https://github.com/wdhudiekou/IRFS)

### TIMFusion: A Task-guided, Implicitly-searched and Meta-initialized Deep Model for Image Fusion

IEEE TPAMI, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=A+Task-guided%2C+Implicitly-searched+and+Meta-initialized+Deep+Model+for+Image+Fusion) [![Code][badge-code]](https://github.com/LiuZhu-CV/TIMFusion) [![Python][badge-python]](https://github.com/LiuZhu-CV/TIMFusion)

### CAF: Where Elegance Meets Precision: Towards a Compact, Automatic, and Flexible Framework for Multi-Modality Image Fusion and Applications

IJCAI, 2024.

[![Paper][badge-paper]](https://scholar.google.com/scholar?q=Where+Elegance+Meets+Precision%3A+Towards+a+Compact%2C+Automatic%2C+and+Flexible+Framework+for+Multi-Modality+Image+Fusion+and+Applications) ![No Code][badge-no-code]

---

## 10. Datasets

| Dataset   | Main Use                            |         Annotation | Link                                                         |
| --------- | ----------------------------------- | -----------------: | ------------------------------------------------------------ |
| TNO       | Classic IVIF / night vision         |                 No | [Link](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) |
| RoadScene | Road-scene IVIF                     |                 No | [Link](https://github.com/hanna-xu/RoadScene)                |
| VIFB      | IVIF benchmark                      |                 No | [Link](https://github.com/xingchenzhang/VIFB)                |
| MSRS      | Fusion + segmentation / road scene  |                Yes | [Link](https://github.com/Linfeng-Tang/MSRS)                 |
| LLVIP     | Low-light pedestrian detection      |                Yes | [Link](https://bupt-ai-cz.github.io/LLVIP/)                  |
| M3FD      | Fusion + object detection           |                Yes | [Link](https://github.com/JinyuanLiu-CV/TarDAL)              |
| MFNet     | Multispectral semantic segmentation |                Yes | [Link](http://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) |
| FMB       | Fusion + segmentation benchmark     |                Yes | [Link](https://github.com/JinyuanLiu-CV/SegMiF)              |
| VF-Bench  | Video fusion benchmark              | Yes/Task-dependent | [Link](https://github.com/RollingPlain/IVIF_ZOO)             |

---

## 11. Evaluation Metrics

### 11.1 Image-level Fusion Metrics

| Metric | Meaning                                           |
| ------ | ------------------------------------------------- |
| EN     | Entropy / information quantity                    |
| MI     | Mutual information between source and fused image |
| SSIM   | Structural similarity                             |
| SD     | Standard deviation / contrast                     |
| AG     | Average gradient                                  |
| SF     | Spatial frequency                                 |
| VIF    | Visual information fidelity                       |
| Qabf   | Edge information preservation                     |
| SCD    | Sum of correlations of differences                |
| CC     | Correlation coefficient                           |

### 11.2 Task-level Metrics

| Task                     | Metrics                                     |
| ------------------------ | ------------------------------------------- |
| Object Detection         | mAP, AP50, AP75, Recall, Precision, F1, FPS |
| Semantic Segmentation    | mIoU, mAcc, Pixel Acc                       |
| Salient Object Detection | MAE, F-measure, S-measure, E-measure        |
| Small Target Detection   | Pd, Fa, IoU, nIoU, F1, SCRG                 |
| Efficiency               | Params, FLOPs, FPS, latency, memory         |

### 11.3 Metric Code

[![Metric Code][badge-code]](https://github.com/RollingPlain/IVIF_ZOO/tree/main/Metric)

---

## 12. Suggested Survey Taxonomy

```text
Infrared-Visible Image Fusion
├── Traditional / Optimization-based Fusion
│   ├── Spatial-domain
│   ├── Transform-domain
│   ├── Sparse / Low-rank Representation
│   └── Saliency / Gradient / Total Variation
├── Fusion for Visual Enhancement
│   ├── Auto-Encoder / Encoder-Decoder
│   ├── CNN / Representation Learning
│   ├── GAN
│   ├── Transformer
│   ├── Diffusion / Generative Model
│   ├── Mamba / State Space Model
│   └── Foundation Model / Vision-Language-guided Fusion
├── Data Compatibility
│   ├── Misalignment-robust Fusion
│   ├── Cross-modal Registration
│   ├── Joint Registration-Fusion
│   ├── Noise / Degradation Robustness
│   └── Attack-tolerant Fusion
└── Task Adaptation
    ├── Object Detection-oriented Fusion
    ├── Semantic Segmentation-oriented Fusion
    ├── Salient Object Detection-oriented Fusion
    ├── Video / Event Fusion
    └── Infrared Small Target Detection-oriented Fusion
```

---

## Notes

- `No Code` means no official or stable public code was confirmed in this list.
- Some papers may have unofficial implementations; for a conservative survey list, official code is preferred.
- For a more exhaustive and frequently updated list, check [IVIF_ZOO](https://github.com/RollingPlain/IVIF_ZOO) and [IVIF-Code-Interpretation](https://github.com/liushh39/IVIF-Code-Interpretation).

---

[badge-paper]: https://img.shields.io/badge/Paper-Link-blue
[badge-code]: https://img.shields.io/badge/Code-GitHub-green
[badge-python]: https://img.shields.io/badge/Python-3.x-yellow
[badge-matlab]: https://img.shields.io/badge/MATLAB-Code-orange
[badge-no-code]: https://img.shields.io/badge/Code-None-lightgrey
[badge-more]: https://img.shields.io/badge/More-Info-purple
[badge-github]: https://img.shields.io/badge/GitHub-Repo-black
[badge-survey]: https://img.shields.io/badge/Survey-TPAMI%2FarXiv-blueviolet
[badge-zoo]: https://img.shields.io/badge/IVIF-ZOO-important
