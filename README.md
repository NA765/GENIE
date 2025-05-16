# GENIE

## Introduction

This repository provides code of data construction by HEAP and the score evaluation code for the GENIE benchmark. Below illustrates the data annotation process for HEAP, and each step of the annotation process can be implemented in this code.

![pipeline-edited-v4](https://github.com/user-attachments/assets/f8e8c00b-f15a-4d7e-b89a-a5b12aa688da)


## Code Usage

### Setup

Since the data generation process requires annotation using GPT-4o or other models, you need to set up an API key before using the code. Please replace value of `API_KEY` in [utils/constants.py](utils/constants.py) with your API key.


### Data Construction

The annotation process of HEAP includes low-level annotation, high-level annotation, and combine and structure. First, you need to prepare the synthetic images to be annotated and organize them in the following structure:

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


Then you need to perform low-level error annotation by running the following command:
```
python data_construction/fake_annotation/low_level_error_annotation.py --image_root /path/to/your/generated/images --annotation_path path/to/your/low/level/annotation
```

This allows you to save the results of your low-level annotation into the `annotation_path`, making it convenient for subsequent use.

