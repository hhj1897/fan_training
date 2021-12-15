# FAN Training Code
The PyTorch training code for [FAN](http://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf) \[1\]. It is adapted from the authors' original [Lua version](https://github.com/1adrianb/face-alignment-training) but with several improvements:
1. We improved the "heatmap -> coordinate" inference algorithm. Instead of using `argmax` to get the coordinates, we compute the centroid in each heatmap as the predicted landmark coordinates. This change leads to more accurate results since the output no longer needs to snap to the integer grid. In addition, it also reduces jitter when processing consecutive frames in a video.
2. We improved the data preprocessing and augmentation pipeline in a number of ways:
   1) The groundtruth heatmaps are now generated with sub-pixel sampling so that they are centred at the precise coordinates of the target landmarks (instead of having the coordinates to be snapped to the integer grid first).
    2) We switched to use [albumentations](https://github.com/albumentations-team/albumentations) for data augmentation. This enabled us to easily add more commonly used augmentation steps into the training pipeline.
   3) We changed the pre-processing procedure so that the heatmaps are generated after applying the data augmentation transforms to the landmark coordinates. Because of this, the data loader now runs much faster. In addition,the shape of the generated heatmaps (covariance of the Gaussian) is no longer affected by the augmentation steps.
3. We introduced a normalised landmark distance loss. This was not possible in the original code because `argmax` is not differentiable, but our improved "heatmap -> coordinate" inference algorithm is.

We performed experiments on 300W and 300W-LP. Results show that after all the improvements, our model (with 2 hourglasses) trained on 300W-LP could achieve an __AUC (at 0.07  face size) of 68.3% on 300W test set__, an __1.4% absolute improvement__ to the performance (__66.9%__) claimed (with a network of 4 hourglasses) in the original paper.

The best model we trained has been included in [ibug.face_alignment](https://github.com/hhj1897/face_alignment), with the name "__2dfan2_alt__".

## Prerequisite
`pip install -r requirements.txt`

## How to Use
All experiments can be found in [experiments.ipynb](./experiments.ipynb).

## References
\[1\] Bulat, Adrian, and Georgios Tzimiropoulos. "[How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks).](http://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf)" In Proceedings of the IEEE International Conference on Computer Vision, pp. 1021-1030. 2017.