<div align="center">
<h1>MIM4D </h1>
<h3>Masked Modeling with Multi-View Video for Autonomous Driving Representation Learning</h3>

[Jialv Zou](https://github.com/Doctor-James)<sup>1</sup> \*, [Bencheng Liao](https://github.com/LegendBC)<sup>1,2</sup> \*, [Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>3</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1 :email:</sup>
 
<sup>1</sup>  School of EIC, HUST, <sup>2</sup>  Institute of Artificial Intelligence, HUST,   <sup>3</sup> Horizon Robotics

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.


</div>


#



### News


* **` Mar. 14th, 2024`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️


## Abstract
Learning robust and scalable visual representations from massive multi-view video data remains a challenge in computer vision and autonomous driving. Existing pre-training methods either rely on expensive supervised learning with 3D annotations, limiting the scalability, or
focus on single-frame or monocular inputs, neglecting the temporal information. We propose MIM4D, a novel pre-training paradigm based on dual masked image modeling (MIM). MIM4D leverages both spatial and temporal relations by training on masked multi-view video inputs. It 
constructs pseudo-3D features using continuous scene flow and projects them onto 2D plane for supervision. To address the lack of dense 3D supervision, MIM4D reconstruct pixels by employing 3D volumetric differentiable rendering to learn geometric representations. 
We demonstrate that MIM4D achieves state-of-the-art performance on the nuScenes dataset for visual representation learning in autonomous driving. It significantly improves existing methods on multiple downstream tasks, including BEV segmentation (8.7% IoU), 
3D object detection (3.5% mAP), and HD map construction (1.4% mAP). Our work offers a new choice for learning representation at scale in autonomous driving.


<div align="center">
<img src="assets/architecture.png" />
</div>




## Citation
If you find MIM4D is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
 @article{mim4d,
  title={MIM4D: Masked Modeling with Multi-View Video for Autonomous Driving Representation Learning},
  author={Jialv Zou and Bencheng Liao and Qian Zhang and Wenyu Liu and Xinggang Wang},
  journal={arXiv preprint arXiv:TODO},
  year={2024}
}
```




