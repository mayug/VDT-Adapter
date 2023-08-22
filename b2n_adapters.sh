

## cub dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_cub_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_cub_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_cub_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_cub_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_cub_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_cub_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_cub_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh cub vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_cub_vit_shot16_0.2_b2n.log


## oxford_pets dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh oxford_pets vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_oxford_pets_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh oxford_pets vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_oxford_pets_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh oxford_pets vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_oxford_pets_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh oxford_pets vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_oxford_pets_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh oxford_pets vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_oxford_pets_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh oxford_pets vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_oxford_pets_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh oxford_pets vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_oxford_pets_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh oxford_pets vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_oxford_pets_vit_shot16_0.2_b2n.log

## oxford_flowers dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh oxford_flowers vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_oxford_flowers_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh oxford_flowers vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_oxford_flowers_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh oxford_flowers vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_oxford_flowers_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh oxford_flowers vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_oxford_flowers_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh oxford_flowers vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_oxford_flowers_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh oxford_flowers vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_oxford_flowers_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh oxford_flowers vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_oxford_flowers_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh oxford_flowers vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_oxford_flowers_vit_shot16_0.2_b2n.log


## caltech-101 dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh caltech-101 vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_caltech-101_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh caltech-101 vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_caltech-101_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh caltech-101 vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_caltech-101_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh caltech-101 vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_caltech-101_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh caltech-101 vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_caltech-101_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh caltech-101 vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_caltech-101_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh caltech-101 vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_caltech-101_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh caltech-101 vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_caltech-101_vit_shot16_0.2_b2n.log


## ucf101 dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh ucf101 vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_ucf101_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh ucf101 vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_ucf101_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh ucf101 vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_ucf101_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh ucf101 vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_ucf101_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh ucf101 vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_ucf101_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh ucf101 vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_ucf101_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh ucf101 vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_ucf101_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh ucf101 vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_ucf101_vit_shot16_0.2_b2n.log

## food-101 dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh food-101 vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_food-101_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh food-101 vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_food-101_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh food-101 vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_food-101_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh food-101 vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_food-101_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh food-101 vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_food-101_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh food-101 vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_food-101_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh food-101 vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_food-101_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh food-101 vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_food-101_vit_shot16_0.2_b2n.log

## stanford_cars dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh stanford_cars vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_stanford_cars_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh stanford_cars vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_stanford_cars_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh stanford_cars vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_stanford_cars_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh stanford_cars vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_stanford_cars_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh stanford_cars vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_stanford_cars_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh stanford_cars vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_stanford_cars_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh stanford_cars vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_stanford_cars_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh stanford_cars vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_stanford_cars_vit_shot16_0.2_b2n.log

## eurosat dataset, best residual_ratio = 0.2, 0.5 for 16shot
bash scripts/clip_adapter/main_gpt.sh eurosat vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_eurosat_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh eurosat vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_eurosat_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh eurosat vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_eurosat_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh eurosat vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_eurosat_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh eurosat vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_eurosat_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh eurosat vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_eurosat_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh eurosat vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.5 self0.2_b2n >> logs/self_eurosat_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh eurosat vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.5 self0.2_b2n >> logs/self_eurosat_vit_shot16_0.2_b2n.log

## fgvc_aircraft dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh fgvc_aircraft vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_fgvc_aircraft_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh fgvc_aircraft vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_fgvc_aircraft_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh fgvc_aircraft vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_fgvc_aircraft_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh fgvc_aircraft vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_fgvc_aircraft_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh fgvc_aircraft vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_fgvc_aircraft_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh fgvc_aircraft vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_fgvc_aircraft_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh fgvc_aircraft vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_fgvc_aircraft_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh fgvc_aircraft vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_fgvc_aircraft_vit_shot16_0.2_b2n.log


## imagenet dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh imagenet vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_imagenet_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh imagenet vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_imagenet_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh imagenet vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_imagenet_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh imagenet vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_imagenet_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh imagenet vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_imagenet_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh imagenet vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_imagenet_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh imagenet vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_imagenet_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh imagenet vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_imagenet_vit_shot16_0.2_b2n.log


## sun397 dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh sun397 vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_sun397_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh sun397 vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_sun397_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh sun397 vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_sun397_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh sun397 vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_sun397_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh sun397 vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_sun397_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh sun397 vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_sun397_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh sun397 vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_sun397_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh sun397 vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_sun397_vit_shot16_0.2_b2n.log

## dtd dataset, best residual_ratio = 0.2
bash scripts/clip_adapter/main_gpt.sh dtd vit_b16_c16_ep10_batch1 end 16 1 False base self_attn 0.2 self0.2_b2n >> logs/self_dtd_vit_shot1_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh dtd vit_b16_c16_ep10_batch1 end 16 1 False new self_attn 0.2 self0.2_b2n >> logs/self_dtd_vit_shot1_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh dtd vit_b16_c16_ep10_batch1 end 16 5 False base self_attn 0.2 self0.2_b2n >> logs/self_dtd_vit_shot5_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh dtd vit_b16_c16_ep10_batch1 end 16 5 False new self_attn 0.2 self0.2_b2n >> logs/self_dtd_vit_shot5_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh dtd vit_b16_c16_ep10_batch1 end 16 10 False base self_attn 0.2 self0.2_b2n >> logs/self_dtd_vit_shot10_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh dtd vit_b16_c16_ep10_batch1 end 16 10 False new self_attn 0.2 self0.2_b2n >> logs/self_dtd_vit_shot10_0.2_b2n.log

bash scripts/clip_adapter/main_gpt.sh dtd vit_b16_c16_ep10_batch1 end 16 16 False base self_attn 0.2 self0.2_b2n >> logs/self_dtd_vit_shot16_0.2_b2n.log
bash scripts/clip_adapter/main_gpt.sh dtd vit_b16_c16_ep10_batch1 end 16 16 False new self_attn 0.2 self0.2_b2n >> logs/self_dtd_vit_shot16_0.2_b2n.log
