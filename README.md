# LMM4Text-KVQA
Official Implementation of our EMLNP 2024 Paper "Visual Text Matters: Improving Text-KVQA with Visual Text Entity Knowledge-aware Large Multimodal Assistant"

[paper](https://anandmishra22.github.io/files/Abhirama-EMNLP24.pdf) | [arxiv]() | [project page](https://vl2g.github.io/projects/LMM4Text-KVQA/)

## Environment
**To setup environment**

We recomment to use the following docker from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).

```
nvcr.io/nvidia/pytorch:23.12-py3
```

```
conda env create -n kalma --file kalma.yml
conda activate kalma
```

## Data
Download the all three splits of Text-KVQA data (Singh et al., ICCV'19) from [here](https://textkvqa.github.io).

## VisTEL

### Preprocessing the data
To run VisTEL, training and inference. The following files are required:

1. Images.
2. Knowledge Base with all the entities.
3. OCR output of every image (to be stored at ```dataset/{split_name}/ocr_output.json```). [OCR pipeline: DBNet + ParSeq. Refer to DBNet implementation [here](https://github.com/MhLiao/DB) and ParSeq implementation [here](https://github.com/baudm/parseq).]
4. Top 5 candidates for each image based on Normalised edit distance scores between the OCR-ed text of the image with the entity names in the knowledge base. (to be stored at ```dataset/{split_name}/filtered_ranked_titles_train.json```)

Set respective paths ```train.json``` and ```test.json``` in the ```configs/vistel_configs/{split_name}/``` folder.

### Training
```
python src/vistel/train.py configs/vistel_configs/{split_name}/train.json
```

### Testing
```
python src/vistel/test.py configs/vistel_configs/{split_name}/test.json
```

### Post processing results
Results should be postprocessed as follows: Text between 'ASSISTANT:' and '[END]' will be the linked entity. Have to be saved at ```dataset/{split_name}/vistel_titles.json```

## KaLMA

### Required data
To run KaLMA, training and inference. The following files are required:

1. Images.
2. Knowledge Base with all the entities.
3. QA files.
4. vistel_titles.json for all splits.

### Training
```
python src/kalma/train.py configs/kalma/{split_name}/train.json
```

### Testing
```
python src/kalma/test.py configs/kalma/{split_name}/test.json
```

### Post processing results
Results should be postprocessed as follows: Text between 'ASSISTANT:' and '[END]' will be the generated answer.

## License
This code and data are released under the [MIT license](LICENSE.txt).

## Cite
If you find this data/code/paper useful for your research, please consider citing.

```
@inproceedings{retvqa,
  author       = {Abhirama Subramanyam Penamakuri and
                  Anand Mishra},
  title        = {Visual Text Matters: Improving Text-KVQA with Visual Text Entity Knowledge-aware Large Multimodal Assistant},
  booktitle    = {EMNLP},
  year         = {2024},
}
```

## Acknowledgements
1. We used code-base and pre-trained models of [LLaVA](https://huggingface.co/docs/transformers/en/model_doc/llava), [DBNet](https://github.com/MhLiao/DB), [ParSeq](https://github.com/baudm/parseq).
2. Abhirama S. Penamakuri is supported by Prime Minister Research Fellowship (PMRF), Minsitry of Education, Government of India.
3. This work was partly supported by the IIT Jodhpur Seed Research Grant and National Language Translation Mission (NLTM): Bhashini project by the MeitY, Government of India.
