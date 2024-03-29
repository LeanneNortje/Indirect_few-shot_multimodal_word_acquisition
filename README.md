# Visually grounded few-shot word learning in low-resource settings: Indirect few-shot MattNet

This repo provides the source code for the MattNet model that uses the indirect method to do few-shot classification and retrieval.
I.e. MattNet not fine-tuned on the mined few-shot pairs.
The paper which is accepted to TASLP in 2024, is only available [here](https://arxiv.org/abs/2306.11371).

## Important note:

The following instructions should be followed within each of the model folders. For example follow these steps in the directory ```100-shot_5-way```

## Data

Copy the ```support_set``` folder from [here](https://github.com/LeanneNortje/Mulitmodal_few-shot_word_acquisition.git).
Download the MSCOCO data [here](https://cocodataset.org/#download). We used the 2014 splits, but all the image samples can be taken from the 2017 splits as well. Just replace the ```train_2014``` and ```val_2014``` in the image names with ```train_2017``` and ```val_2017```.

## Preprocessing

```
cd preprocessing/
python sample_lookups.py /path/to/SpokenCOCO/
python preprocess_spokencoco_dataset.py
cd ../
```

## Using pretrained model weights

If you want to use the model checkpoints, download the checkpoints given in the release and move the model_metadata folder to the model directory.
Take care to follow the exact directory layout given here:

```bash
├── model_metadata
│   ├── <model_name>
│   │   ├── <model_instance>
│   │   │   ├── models
│   │   │   ├── args.pkl
│   │   │   ├── params.json
│   │   │   ├── training_metadata.json
```

## Model training

First download the ```pretrained``` weights to initialise the model in the releases and extract it in the project directory as follows:

```bash
├── pretrained
│   ├── best_ckpt.pt
│   ├── last_ckpt.pt
```

## Evaluation

To do few-shot classification:
```
python few-shot_classification.py
```

To do few-shot retrieval:
```
python few-shot_retrieval.py
```
