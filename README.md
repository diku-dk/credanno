# cRedAnno 🤏

[[`arXiv`]()] [[`Dataset`](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)]

**C**onsiderably **Red**ucing **Anno**tation Need in Self-Explanatory Models

------


## Performance overview

<div align="center">
<figure align="center">
    <img src="./imgs/anno_reduce.svg" alt="anno_reduce" width="40%" />
    <figcaption>Annotation reduction performance</figcaption>
</figure>
</div>

<table align="center" style="margin: 0px auto; text-align:center; vertical-align:middle" >
<thead>
  <tr>
    <th rowspan="2"></th>
    <th colspan="7" style="text-align:center;">Nodule attributes</th>
    <th rowspan="2">Malignancy</th>
  </tr>
  <tr>
    <th>Sub</th>
    <th>Cal</th>
    <th>Sph</th>
    <th>Mar</th>
    <th>Lob</th>
    <th>Spi</th>
    <th>Tex</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="9" style="text-align:left;">Full annotation</td>
  </tr>
  <tr>
    <td style="text-align:left;">cRedAnno (50-NN)</td>
    <td>94.93</td>
    <td>92.72</td>
    <td>95.58</td>
    <td>93.76</td>
    <td>91.29</td>
    <td>92.72</td>
    <td>94.67</td>
    <td style="text-align:center;">87.52</td>
  </tr>
  <tr>
    <td style="text-align:left;">cRedAnno (250-NN)</td>
    <td>96.36</td>
    <td>92.59</td>
    <td>96.23</td>
    <td>94.15</td>
    <td>90.90</td>
    <td>92.33</td>
    <td>92.72</td>
    <td style="text-align:center;">88.95</td>
  </tr>
  <tr>
    <td style="text-align:left;">cRedAnno (trained)</td>
    <td>95.84</td>
    <td>95.97</td>
    <td>97.40</td>
    <td>96.49</td>
    <td>94.15</td>
    <td>94.41</td>
    <td>97.01</td>
    <td style="text-align:center;">88.30</td>
  </tr>
  <tr>
    <td colspan="9" style="text-align:left;">Partial annotation</td>
  </tr>
  <tr>
    <td style="text-align:left;">cRedAnno (10%, 50-NN)</td>
    <td>94.93</td>
    <td>92.07</td>
    <td>96.75</td>
    <td>94.28</td>
    <td>92.59</td>
    <td>91.16</td>
    <td>94.15</td>
    <td style="text-align:center;">87.13</td>
  </tr>
  <tr>
    <td style="text-align:left;">cRedAnno (10%, 150-NN)</td>
    <td>95.32</td>
    <td>89.47</td>
    <td>97.01</td>
    <td>93.89</td>
    <td>91.81</td>
    <td>90.51</td>
    <td>92.85</td>
    <td style="text-align:center;">88.17</td>
  </tr>
  <tr>
    <td style="text-align:left;">cRedAnno (1%, trained) 🤏</td>
    <td>91.81</td>
    <td>93.37</td>
    <td>96.49</td>
    <td>90.77</td>
    <td>89.73</td>
    <td>92.33</td>
    <td>93.76</td>
    <td style="text-align:center;">86.09</td>
  </tr>
</tbody>
</table>


## Usage instruction

### Dependencies

Create an environment from the [`environment.yml`](./environment.yml) file:
```bash
conda env create -f environment.yml
```
and install [`pylidc`](https://pylidc.github.io/) for dataset pre-processing.

### Data pre-processing

Use [`extract_LIDC_IDRI_nodules.py`](./extract_LIDC_IDRI_nodules.py) to extract nodule slices. 

### Training

#### Unsupervised feature extraction

Following [`DINO`](https://github.com/facebookresearch/dino), to train on the extracted nodules:

```bash
python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /path_to_extracted_dir/Image/train --output_dir ./logs/vits16_pretrain_full_2d_ann --epochs 300
```

The reported results start from the ImageNet-pretrained full weights provided for [`ViT-S/16`](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth), which should be put under `./logs/vits16_pretrain_full_2d_ann/`.

#### Supervised prediction

To train the predictors:

```bash
python eval_linear_joint.py --pretrained_weights ./logs/vits16_pretrain_full_2d_ann/checkpoint.pth --data_path /path_to_extracted_dir --output_dir ./logs/vits16_pretrain_full_2d_ann --label_frac 0.01
```

or use the k-NN classifiers:

```bash
python eval_knn_joint.py --pretrained_weights ./logs/vits16_pretrain_full_2d_ann/checkpoint.pth --data_path /path_to_extracted_dir --output_dir ./logs/vits16_pretrain_full_2d_ann --label_frac 0.01
```

In both cases, `--label_frac` controls the used fraction of annotations.

The results are saved in `pred_results_*.csv` files under specified `--output_dir`.



## Code reference

Our code adapts from [`DINO`](https://github.com/facebookresearch/dino).



