from PIL import Image
import requests
import torch
import json
import sys
from tqdm import tqdm
import time
import wandb
import random

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# import accelerator
from accelerate import Accelerator

from transformers import AutoProcessor, LlavaForConditionalGeneration, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel

from accelerate import DistributedDataParallelKwargs


# set seed
torch.manual_seed(42)

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_8bit=True)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")


config_file_path = sys.argv[1]
with open(config_file_path, 'r') as f:
    config = json.load(f)


if config['use_gpu']:
    device = 'cuda'
    # model.to(device)
else:
    device = 'cpu'
    model.to(device)

wandb.init(project=config['wandb_project'], config=config)

model = PeftModel.from_pretrained(model, config['checkpoint_path'])


test_file = config['test_file']

# Read data from files
def read_data(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data

test_data = read_data(test_file)
kb = read_data(config['kb_file'])
ocr = read_data(config['ocr_output_path'])


class TKVQADataset(Dataset):
    def __init__(self, data, processor, config):
        super().__init__()
        self.raw_data = data
        self.processor = processor
        self.config = config
        self.data = []

        candidate_entity_file = self.config['candidate_entity_file']
        with open(candidate_entity_file, 'r') as f:
            self.candidate_entities = json.load(f)

        # all_images 
        all_images = {}
        for _, value in self.raw_data.items():
            img_name = value['image_path'].split('/')[-1]
            if img_name not in all_images:
                if img_name.split('.')[0] in self.candidate_entities.keys():
                            all_images[img_name] = value['image_path']


        print(f'No of samples: {len(list(all_images.keys()))}')

        for key, value in all_images.items():
            temp = {}
            
            # candidate entites, additional context OCR and answer
            candidate_entities = self.candidate_entities[key.split('.')[0]]
            random.shuffle(candidate_entities)
            candidate_entity_string = ''
            for id, each_candidate_entity in enumerate(candidate_entities):
                candidate_entity_string += f'{id+1}. "{kb[each_candidate_entity]["has title"].lower()}"\n'

            # get the OCR
            if key in ocr.keys():
                ocr_text = ocr[key]
                # print(f'OCR: {ocr_text}')
            else:
                ocr_text = ''
                # print('OCR: Not available')

            # get the answer
            answer = kb[key.split('.')[0]]['has title'].lower()
            # print(f'Answer: {answer}')

            temp['image_path'] = value
            temp['candidate_entity_string'] = candidate_entity_string
            temp['ocr_text'] = ocr_text
            temp['answer'] = answer

            self.data.append(temp)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # added a shortcut in the next line change it later #TODO
        image = Image.open(item['image_path'])

        # prompt creation
        prompt_with_answer = f"<image>\nUSER: Given an image. The task is to link the visual text {item['ocr_text']} to one of the following entities: {item['candidate_entity_string']}.\nASSISTANT: </s> "
        # print(prompt_with_answer)
        inputs = self.processor(text=prompt_with_answer, images=image, return_tensors="pt", padding='max_length', max_length=800, truncation=True)

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        img_id = item['image_path'].split('/')[-1].split('.')[0]

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }

        return inputs, img_id
    
test_dataset = TKVQADataset(test_data, processor, config)
test_dataloader = DataLoader(test_dataset, batch_size=config['valid_batch_size'], shuffle=True)


pbar = tqdm(test_dataloader, desc=f"Test Loss: 0.0000", dynamic_ncols=True)
results = {}
for batch_idx, batch in enumerate(pbar):

    inputs, q_ids = batch

    inputs = {k:v.to(device) for k,v in inputs.items()}

    generate_ids = model.generate(**inputs, max_new_tokens=100)
    text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    for idx, q_id in enumerate(q_ids):
        results[q_id] = text[idx]

    if batch_idx % 10 == 0:
        # write results to results/scene/results.json
        with open(config['results_save_path'], "w") as f:
            json.dump(results, f, indent=4)

# write results to results/scene/results.json
with open(config['results_save_path'], "w") as f:
    json.dump(results, f, indent=4)