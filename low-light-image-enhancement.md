### 2020
- Self-supervised Image Enhancement Network: Training with Low Light Images Only
\[[Code](https://github.com/hitzhangyu/Self-supervised-Image-Enhancement-Network-Training-With-Low-Light-Images-Only)\]  
`light enhancement` `self-supervised` `retinex` `sRGB`  
该方法基于Retinex模型，使用网络预测预测光照图和反射系数，根据Retinex成像公式构建重建损失、反射系数梯度加权的光照图的TV损失、直方图均衡化后图像与反射系数之间的绝对误差、反射系数梯度的L1正则化。
- Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
\[[Code](https://github.com/Li-Chongyi/Zero-DCE)\]  
`CVPR` `light enhancement` `self-supervised` `sRGB`  
使用输入图预测8组三通道光照增强系数，系数维度与输入图像完全相同，即每个像素都有8组非线性变换参数；论文中使用了空间一致性损失(保持增强前后像素强度与其邻域像素强度的差异不变)、曝光控制损失(16x16局部区域平均像素强度与完美曝光等级的绝对差，论文发现设定E从0.4到0.7的效果无明显差异)、色彩恒常性损失(根据灰色世界假说，即认为场景中RGB三色的均值应该是相同的，该方法可以消除环境光的影响)、光照平滑损失(正则非线性变换系数梯度，即让各通道内的系数趋于相同，即一定程度保持原始图像中像素强度间的相对差异)。
- Learning to Restore Low-Light Images via Decomposition-and-Enhancement  
`CVPR` `light enhancement` `denoising` `sRGB`  
现有的大部分方法在对低光增强时都假设图片时无噪声的，导致的结果就是图片中的噪声也被增强了。作者认为在图像的低频分量中去噪更容易，通过两个感受野不一样的卷积差来过滤高低频分量；其实最关键的还是损失函数部分，预测的有噪声低频图像与用引导滤波处理过的GT间的损失(迫使网络对低频分量进行去噪增强)、最后重建图像与GT间的损失(精修一下)。
- From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement
\[[Code](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)\]  
`CVPR` `light enhancement` `denoising` `semi-supervised` `sRGB`  
作者认为用带有GT的低光增强数据训练模型可以达到抑制噪声、修复细节的功能，无监督又可以制作更大的数据集使模型适用性更强，可以恢复颜色、光照、对比度更好。因此提出了一个递归形式的U-Net残差网络，使用有监督数据训练，以达到去噪、增强的效果，为了进一步改善效果，又加入一个GAN网络，使用无监督数据进一步提升图片质量。
- CycleISP: Real Image Restoration via Improved Data Synthsis
\[[Code](https://github.com/swz30/CycleISP)\]  
`CVPR` `denoising` `RAW` `sRGB` `synthetic data`  
sRGB图像的去噪使用的合成噪声大部分使用的都是加性高斯白噪声，而相机成像噪声在RAW空间是与像素强度有关的噪声，经过ISP之后得到的sRGB就更不可能是简单的高斯噪声了。该论文使用两个模块分别建模正向RAW->RGB和反向RGB->RAW两个过程，两个模块的训练都需要成对的(RGB,RAW)数据，模型训练好之后就可以用无噪声图片合成有噪声图片，再用这些合成数据有监督的训练去噪模型。对于sRGB数据，输入是RGB图像，网络预测噪声；对于RAW数据，输入是4通道图像+4通道noise level map，输出同样是噪声估计值。
- Transfer Learning from Synthetic to Real-Noise Denoising with Adaptive Instance Normalization
\[[Code](https://github.com/terryoo/AINDNet)\]  
`CVPR` `denoising` `synthetic data` `sRGB`  
方法与CBDNet一致，通过使用down/up-sampling增加感受野，提高噪声方差估计性能，对应的非对称损失也改成了多尺度的，交替训练改成了合成数据训练再用真实数据迁移训练的过程。
- Attention-Based Network For Low-Light Image Enhancement
\[[Code](https://github.com/Justones/ALEN)\]  
`light enhancement` `denoising` `RAW`  
该方法整体与SID一致，只不过网络结构中引入了none-local attention操作，损失函数多引入了一个SSIM。输入数据不再是像SID那样提前乘一个放大系数，而是分别乘\[0.5,0.8,1.0,1.2\](而代码中用的系数是\[0.8,1.0,1.2,1.5\])得到四种不同亮度图片作为网络输入。
- Burst Denoising of Dark Images \[[Code](https://github.com/hucvl/dark-burst-photography)\]  
`denoising` `RAW`  
为了进一步改善SID方法存在的模糊、极低光效果不好的问题，论文提出coarse-to-fine结构进行改进。先把输入图降采样(1/2)，训练一个降采样的去噪模型(coarse)，再上采样去噪图像与输入图大小相同，计算得到一个残差，把以上三个图全部送入下一个去噪网络(fine)学习再次的去噪模型。
- Attention Guided Low-light Image Enhancement with a Large Scale Low-light Simulation Dataset
\[[Code](https://github.com/yu-li/AGLLNet)\]  
`IJCV` `light enhancement` `denoising` `synthetic data`  

### 2019
- EnlightenGAN: Deep Light Enhancement without Paired Supervision
\[[Code](https://github.com/VITA-Group/EnlightenGAN)\]  
`light enhancement` `sRGB`
- Noise2Void - Learning Denoising from Single Noisy Images
\[[Code](https://github.com/juglab/n2v)\]  
`CVPR` `denoising`  
假设图像中像素之间不是相互独立的、噪声是与像素相关但相互独立的，同样假设是零均值噪声。既然像素之间是有联系的，那么就是用像素周围的像素重建自身就是Noise2Void的核心思想了，这样就只需要噪声图片就行了。
- Noise2Self: Blind Denoising by Self-Supervision
\[[Code](https://github.com/czbiohub/noise2self)\]  
`denoising`  
与Noise2Void类似，只不过BlindSpot那个点使用周围像素的均值替换的，没发现什么新奇的东西。
- High-Quality Self-Supervised Deep Image Denoising
\[[Code](https://github.com/NVlabs/selfsupervised-denoising)\]
- Learning Raw Image Denoising with Bayer Pattern Unification and Bayer Preserving Augmentation
- Toward Convolutional Blind Denoising of Real Photographs
\[[Code](https://github.com/GuoShi28/CBDNet)\]  
`CVPR` `denoising` `sRGB` `synthetic data`  
考虑到sRGB图像在RAW空间有泊松-高斯噪声，通过ISP以及JPEG压缩之后进一步使噪声变得更复杂，因此论文提出先用一个小网络根据输入图片预测noise level estimation，然后再把它与输入图片拼接在一起输入到去噪网络中得到去噪后的图像。根据已有文献表明，噪声方差估计如果小于真实噪声方差(under-estimation)，那么去噪就不够干净，还会有明显的噪声存在，而估计的大一些也不会带来太多的损失，因此作者提出对噪声方差估计增加一个非对称损失(asymmetric loss)，即如果某个像素点上的方差估计小了，那么就用一个更大的系数放大这个损失即可，为了保持噪声方差估计的平滑性，还对其算一个TV loss。合成数据首先把sRGB图像逆向ISP过程到RAW，在RAW上添加噪声，然后再通过ISP过程回到sRGB。训练时真实数据与合成数据交替使用，由于真实数据不知道噪声方差，所以只是用TV loss和重建损失。
- Unprocessing Images for Learned Raw Denoising
\[[Code](https://github.com/google-research/google-research/tree/master/unprocessing)\]  
`CVPR` `denoising` `RAW` `synthetic data`  
该方法先建模RAW到sRGB的过程，该过程不包含可学习参数，仅仅是对sRGB图像的一种反向操作，然后再对得到的RAW图像添加shot和read噪声，最后通过去噪网络得到去噪的RAW。网络输入是反向操作得到的RAW图像+对应的noise level estimation，输出是去噪后的RAW图像，损失函数是再转换到sRGB下的L1损失。

### 2018
- Learning to See in the Dark
\[[Code](https://github.com/cchen156/Learning-to-See-in-the-Dark)\] `Dataset`  
`light enhancement` `denoising` `RAW`  
论文收集了一批低光照图像和对应的长曝光图像，网络输入是低光照有噪声图像，输出是三通道RGB图像。该方法的目的就是取代传统ISP的过程，作者发现使用libraw将GT图片从RAW转换成RGB时不做直方图均衡化得到的效果更好，但是图片亮度偏暗一些，此时再进行直方图均衡化效果更佳。输入图片会根据与GT曝光时间差异乘上[100,250,300]，即事先把噪声图提亮，然后给网络去噪，该网络的局限性就是需要人为去设定放大系数，因为实际使用时我们并没有GT的曝光时间信息。
- Deep Retinex Decomposition for Low-Light Enhancement
- Noise2Noise: Learning Image Restoration without Clean Data
\[[Code](https://github.com/NVlabs/noise2noise)\]  
`denoising`  
对于L2损失而言，最小化某个样本的多次观测损失时，其最终的效果是多次观测的均值；基于这个假设，作者进一步分析，如果噪声是0均值的，那么对真实值是加另一个噪声后，并不会改变最终网络的效果，因为从统计上将加过噪声之后的真实数据的均值是没有发生变化的，这也就是论文所说的Noise2Noise，即不是需要(noisy,GT)对，而是需要(noisy,noisy)数据。该方法对噪声是零均值的假设太强了，缺乏实际用处。
- Deep Image Prior \[[Code](https://github.com/DmitryUlyanov/deep-image-prior)\]  
`CVPR` `denoising`  
作者发现用网络去拟合正常的照片网络时loss下降的很快，而去拟合非正常的照片时网络表现出了抗拒的作用，虽然经过长时间的迭代网络还是会去拟合，但是如果我们控制网络迭代的步数，那么网络是不是就只拟合了正常的部分，而去除了图片中异常的部分？这就是标题名所说的Prior，作者通过实现证明这种先验在控制一定的迭代步数时确实有用。首先随机初始化一个网络权重，输入是一个随机生成之后固定的噪声，用这个噪声去生成我们的目标图片，最简单的loss就是L2，如果是有噪声的图片，随着优化网络权重的进行，生成的图片却是有去燥效果的图片！
- Burst Denoising with Kernel Prediction Networks
- A High-Quality Denoising Dataset for Smartphone Cameras
\[[PDF](http://www.cse.yorku.ca/~mbrown/pdf/sidd_cvpr2018.pdf)\] `Dataset`
- A Dataset for Real Low-Light Image Noise Reduction \[[PDF](https://arxiv.org/pdf/1409.8230.pdf)\] `Dataset`

### 2016
- LIME: Low-Light Image Enhancement via Illumination Map Estimation
- Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
- A Holistic Approach to Cross-Channel Image Noise Modeling and its Application to Image Denoising

### 2008
- Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data
\[[PDF](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.1943&rep=rep1&type=pdf)\]
