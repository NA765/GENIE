# GENIE

## Introduction

This repository provides code of data construction by HEAP and the score evaluation code for the GENIE benchmark. Below illustrates the data annotation process for HEAP, and each step of the annotation process can be implemented in this code.

![pipeline-edited-v4](https://github.com/user-attachments/assets/f8e8c00b-f15a-4d7e-b89a-a5b12aa688da)


## Code Usage

### Setup

Since the data generation process requires annotation using GPT-4o or other models, you need to set up an API key before using the code. Please replace value of `API_KEY` in [utils/constants.py](utils/constants.py) with your API key.

***Note**: If you have acess to Azure OpenAI platform with corresponding models implemented, you can replace the function `gpt4o_response` in [utils/gpt4o.py](utils/gpt4o.py) with `gpt4o_response_legacy` in this file.



### Generation of Synthetic Images (Optional)

If you want to generate images yourself, you can run [data_construction/image_genrate.py](data_construction/image_genrate.py) first:

```
python data_construction/image_genrate.py --save_image_root ./generated_images --images_per_cat 4
```

You can also add or remove `model_ids` and `prompts` in this file to control types of generated images. The final structure of image folder will be in the following structure:

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


### Data Construction of Generated Images


The annotation process of HEAP includes low-level annotation, high-level annotation, and combine and structure. Before the full annotation process, you need to organize the synthetic images in the structure mentioned in Generation of Synthetic Images part.



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
python data_construction/fake_annotation/annotation_combine.py --input_folder /path/to/your/generated/images --low_level_folder path/to/your/low/level/annotation --high_level_folder path/to/your/revised/high/level/annotation --output_folder path/to/your/final/fake/annotation
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
python data_construction/real_annotation/annotation_real_combine.py --input_folder /path/to/your/real/images --high_level_folder path/to/your/high/level/annotation --output_folder path/to/your/final/real/annotation
```

The final annotation will be saved to `output_folder`.

### Save with JSON Format

If you want to package the final annotation results into a JSON format similar to the LLaVA training data format (refer [here](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)), you can run the following command:

```
python data_construction/final_json_create.py --fake_image_root /path/to/your/generated/images --fake_annotation_root path/to/your/final/fake/annotation --real_image_root /path/to/your/real/images --real_annotation_root path/to/your/final/real/annotation --output_combined_json /path/to/your/output/json/file
```

You can finally save the results to `output_combined_json` and start training!🎉


### Manual Revison for High-level Error Annotation Stage 2

If you want to implement manual revision during the revision stage of high-level error annotation for synthetic images, you can deploy Label Studio to facilitate the process. Follow the instructions below to complete this process.

#### Setup

First, you need to fill the corresponding content in [utils/constants.py](utils/constants.py). Replace `SERVER_IP` with your IP, and `IMAGE_ROOT` for your generated image root, `ANNOTATION_ROOT` for your generated annotation root.

#### Create JSON file for Labeling Tasks

Then you need to create JSON file for labeling tasks in Label Studio. Run the following command to obtain JSON file in `output_path`:

```
python data_construction/manual_annotation.py --image_dir /path/to/your/generated/images --text_dir path/to/your/unrevised/high/level/annotation --output_path /path/to/your/output/json/file
```

#### Initialize Your Label Studio Server

To ensure your image and annotation will be correctly loaded in your server, run this command first:

```
python data_construction/manual_annotation/label_studio_server_init.py
```

Then run the following command to start server:

```
label-studio start --port 8080
```

This will create a label server on [http://127.0.0.1:8080/projects](http://127.0.0.1:8080/projects), you can start labeling by sign in or sign up an account.

Then you can create label tasks on your own. Click the `Create` button on the top right corner, and fill the description of this project in `Project Name` column. Next, **upload the json file for labeling tasks in the `Data Import` column**. When this is ready, choose `Labeling Setup` column and click `Custom template` on the left, **fill the template with content in [data_construction/manual_annotation/label_studio_template.xml](data_construction/manual_annotation/label_studio_template.xml)**. Finally, click `Save` to save all settings.

Then you can start labeling the tasks!🎉 Below is the screenshot of the Label Studio platform interface for manual annotation. The top-right corner allows adding human revisions, while the bottom-right corner enables adding or removing generated annotation points.
![benchmark_screenshot](https://github.com/user-attachments/assets/7d80074f-f3b1-483f-88cd-0fdf099f3c8d)

When you complete the manual annotation task, click the `Export` button on the project page and select the `JSON` format for export. Finally, run the command below to finish the manual revision process:

```
python data_construction/manual_annotation/annotation_high_level_manual_refine.py --json_path /path/to/your/manual/label/json/path --image_root /path/to/your/generated/images --annotation_root path/to/your/manual/revised/high/level/annotation
```

Now you have finished all manual revision process!🎉


### Evaluation

This repository also provides evaluation code to calculate four metrics: **accuracy**, **match score**, **richness score**, and **hallucination rate**. The following figure illustrates the calculation process of these evaluation metrics.

![image](https://github.com/user-attachments/assets/82b38c4b-7f91-409c-8788-f666d2f056f2)


Before evaluation, you should prepare the output results and ground truth in following JSON format:

```
[
    {
        "image_path": /path/to/your/image,
        "ground_truth": ground_truth_annotation,
        "generated": generated_annotation,
        "label": 0
    },
    ...
]
```

For `"label"`, 0 represents for real and 1 represents for fake. It should be noted that do not change the structure of ground truth and generated results, e.g. do not delete or remove signals like `<begin_of_point>` in the annotations. Then run following command to get the evaluation results:

```
python eval/score_compute.py --annotation_file /path/to/json/file --metrics sentence_transformer
```

`metrics` can be chosen from `sentence_transformer`, `bleu@1`, `bleu@2`, `bleu@3`, `bleu@4`, `rouge`, `meteor` and `gpt_4o`. The results will also be saved to `tmp_eval_result.txt` by default.


## TODO

- [ ] `requirements.txt`
- [x] `Evaluation`
- [ ] `README.md`
- [x] `Code of synthetic image generation`




