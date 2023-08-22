# Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts

This repo contains the code as well as auxiliary data for our ICCV-W paper ['Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts'](https://arxiv.org/abs/2307.11661)


The GPT-4 generated dataset is available in the folder gpt4_data as {dataset_name}.pt files. 



## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run
For the ZS experiments use the following script.

```bash
bash scripts/clip/main_gpt.sh cub vit_b16_c16_ep10_batch1 all zs_gpt_v
```
Arguments are dataset_name, encoder_config, class-sampling(base, new or all classes), exp_name.

For the few-shot experiments use the following script

```bash
bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n_3-5
```

Arguments are dataset_name, encoder_config, coop1,coop2, n_shot, coop3, class-sampling(base, new or all classes), adapter-type, residual-ratio,  exp_name. Arguments coop1-3 are from older code and not used in clip_adapter_gpt.py

main.sh is the script for running default clip adapter.

Please refer to b2n_adapters.sh for the scripts for all shots and all datasets (with tuned residual ratio) for CLIP-A-self in the base 2 new setting.

Residual ratio has to be tuned for each dataset/shot setting

## Citation
If you use this code in your research, please kindly cite our paper.

```bash
@article{maniparambil2023enhancing,
  title={Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts},
  author={Maniparambil, Mayug and Vorster, Chris and Molloy, Derek and Murphy, Noel and McGuinness, Kevin and O'Connor, Noel E},
  journal={arXiv preprint arXiv:2307.11661},
  year={2023}
}
```


This code-base is built on top of [CoOP](https://github.com/KaiyangZhou/CoOp) and [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter).