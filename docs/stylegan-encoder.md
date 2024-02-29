### 2022
- AE-StyleGAN: Improved Training of Style-Based Auto-Encoders
\[[论文解读](https://mp.weixin.qq.com/s/4izPf1UhIvpq5UBS0Hk6-g)\]
\[[PDF](https://arxiv.org/pdf/2110.08718.pdf)\] \[[Code](https://github.com/zideliu/AE-StyleGAN)\]  
`WACV`  
端到端训练Encoder和StyleGAN，与StyleALAE不同的是Encoder输出的编码不会送给判别器，而且StyleALAE还会用生成器生成的图像，增加了训练Encoder的难度，AE-StyleGAN只给判别器看生成的图像和真实图像，效果比StyleALAE好不少。

### 2021
- Designing an Encoder for StyleGAN Image Manipulation
\[[PDF](https://arxiv.org/pdf/2102.02766.pdf)\] \[[Code](https://github.com/omertov/encoder4editing)\]  
`SIGGRAPH` `e4e`  
为了让Encoder有可编辑性，Encoder除了预测出一个w之外，还预测出额外的N个latent code的残差，与w相加之后组成w+空间输入到StyleGAN中。Encoder学习的目标还是重建输入图，使用LPIPS计算重建损失，为了让w+空间不偏离原本的空间，对delta w增加L2正则，还增加了对w+空间的对抗损失，正样本是w，负样本是w+。

- Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation
\[[论文解读](https://mp.weixin.qq.com/s/qj0EdHdJHnzo05XiQZ_7sA)\]
\[[PDF](https://arxiv.org/pdf/2008.00951.pdf)\] \[[Code](https://github.com/eladrich/pixel2style2pixel)\]  
`CVPR` `pSp`  
增加一个ResNet结构的Encoder，把中间层不同分辨率的特征映射到StyleGAN w+空间中，重建损失有LPIPS、L2、identity损失，还增加了Encoder编码的w+ code与StyleGAN自身latent code均值间的L2正则损失。
- ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinemet
\[[PDF](https://arxiv.org/pdf/2104.02699.pdf)\] \[[Code](https://github.com/yuval-alaluf/restyle-encoder)\]  
`ICCV`  
该方法Encoder部分与pSp类似，只不过Encoder输出的重建图像与输入图像间的latent code差值，网络通过多次迭代预测得到最终的latent code，论文中使用N=5次迭代，编辑图像部分采用了e4e类似的方法。