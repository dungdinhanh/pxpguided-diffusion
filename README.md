# PixeAsParam: A Gradient View on Diffusion Sampling with Guidance

This is the codebase for [PixelAsParam: A Gradient View on Diffusion Sampling with Guidance](https://proceedings.mlr.press/v202/dinh23a/dinh23a.pdf).

This repository is based on [openai/improved-diffusion](https://github.com/openai/guided-diffusion), with modifications for PixelAsParam sampling.

# Download pre-trained models

We have utilized the checkpoints from [openai/improved-diffusion](https://github.com/openai/guided-diffusion). Before using these models, please review the corresponding [model card](model-card.md) to understand the intended use and limitations of these models.



Here are the download links for each model checkpoint (Mostly from OpenAI except Diffusion Model for ImageNet64x64):
 * 64x64 unconditional diffusion: (Release soon) 
 * 64x64 classifier: [64x64_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt)
 * 64x64 diffusion: [64x64_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt)
 * 128x128 classifier: [128x128_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_classifier.pt)
 * 128x128 diffusion: [128x128_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt)
 * 256x256 classifier: [256x256_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt)
 * 256x256 diffusion: [256x256_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)
 * 256x256 diffusion (not class conditional): [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)

# Sampling from pre-trained models

To sample from these models, you can use the `classifier_sample.py`, `image_sample.py`, and `super_res_sample.py` scripts.
Here, we provide flags for sampling from all of these models.
We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

For these examples, we will generate 100 samples with batch size 4. Feel free to change these values.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
```

## Classifier guidance

Note for these sampling runs that you can set `--classifier_scale 0` to sample from the base diffusion model.
You may also use the `image_sample.py` script instead of `classifier_sample.py` in that case.

 * 64x64 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS
```

 * 128x128 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt --model_path models/128x128_diffusion.pt $SAMPLE_FLAGS
```

 * 256x256 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS
```

 * 256x256 model (unconditional):

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS
```





# Results

This table summarizes our ImageNet results for pure guided diffusion models:

| Dataset          | FID  | Precision | Recall |
|------------------|------|-----------|--------|
| ImageNet 64x64   | 2.07 | 0.74      | 0.63   |
| ImageNet 128x128 | 2.97 | 0.78      | 0.59   |
| ImageNet 256x256 | 4.59 | 0.82      | 0.52   |
| ImageNet 512x512 | 7.72 | 0.87      | 0.42   |

