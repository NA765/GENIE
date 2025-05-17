# GENIE

## Introduction

This repository provides code of data construction by HEAP and the score evaluation code for the GENIE benchmark. Below illustrates the data annotation process for HEAP, and each step of the annotation process can be implemented in this code.

![pipeline-edited-v4](https://github.com/user-attachments/assets/f8e8c00b-f15a-4d7e-b89a-a5b12aa688da)


## Code Usage

### Setup

Since the data generation process requires annotation using GPT-4o or other models, you need to set up an API key before using the code. Please replace value of `API_KEY` in [utils/constants.py](utils/constants.py) with your API key.



### Generation of Synthetic Images (Optional)


### Data Construction of Generated Images


The annotation process of HEAP includes low-level annotation, high-level annotation, and combine and structure. Before the full annotation process, you need to organize the synthetic images in the following structure:

```
generated_images/
├── source1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── source2/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── source3/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── ...
```

Subfolder name `source1`, `source2` represent the source of generative models, such as `stable-diffusion-v1-5` and `flux.1-dev`. Each subfolder contains synthetic images, with no specific restrictions of number or name.


#### Low-level Error Annotation


Start low-level error annotation by running the following command:
```
python data_construction/fake_annotation/annotation_low_level.py --input_folder /path/to/your/generated/images --output_folder path/to/your/low/level/annotation
```

This allows you to save the results of your low-level annotation into the `output_folder`.


#### High-level Error Annotation

High-level error annotation involves two stages. First, you can obtain the unrevised high-level error annotation by running the following command:

```
python data_construction/fake_annotation/annotation_high_level.py --input_folder /path/to/your/generated/images --output_folder path/to/your/unrevised/high/level/annotation
```

This allows you to save the results of your high-level annotation into the `output_folder`.


After stage 1, run the following command to refine the annotation results:

```
python data_construction/fake_annotation/annotation_high_level_refine.py --image_root /path/to/your/generated/images --annotation_root path/to/your/unrevised/high/level/annotation --output_root path/to/your/revised/high/level/annotation
```

This step will improve the quality of annotation, and revised annotation will be saved to `output_root`.


#### Combine and Restructure

Finally, you need to combine and restructure the final annotation results, which can be achieved by running the following command:

```
python data_construction/fake_annotation/annotation_combine.py --input_folder /path/to/your/generated/images --low_level_folder path/to/your/unrevised/high/level/annotation --high_level_folder path/to/your/revised/high/level/annotation --output_folder path/to/your/final/annotation
```

The final annotation will be saved to `output_folder`.


### Data Construction of Real Images

Before annotating real images, we also need to standardize the folder structure. Please organize the real images in the following structure:

```
real_images/
├── image1.png
├── image2.png
├── image3.png
└── ...
```


The annotation process for real images is simpler compared to synthetic images, as such annotations typically involves restating the plausible components within the image, and thus usually does not involve significant hallucinations. You can obtain the final annotations for real images by running the following two commands:


```
python data_construction/real_annotation/annotation_real.py --input_folder /path/to/your/real/images --output_folder path/to/your/high/level/annotation
python data_construction/real_annotation/annotation_real_combine.py --input_folder /path/to/your/real/images --high_level_folder path/to/your/high/level/annotation --output_folder path/to/your/final/annotation
```

The final annotation will be saved to `output_folder`.

### Save with JSON Format


### Manual Revison for High-level Error Annotation Stage 2



## TODO

- []  `requirements.txt`
- [] `README.md`
- [] `Code of synthetic image generation`




