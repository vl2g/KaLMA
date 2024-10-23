from PIL import Image
import requests
import torch
import json
import sys
from tqdm import tqdm
import time
import wandb

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# import accelerator
from accelerate import Accelerator

from transformers import AutoProcessor, LlavaForConditionalGeneration, get_linear_schedule_with_warmup
from transformers import pipeline, set_seed
from peft import LoraConfig, get_peft_model, TaskType 
from peft import AutoPeftModelForCausalLM, PeftModel

# set seed
torch.manual_seed(42)

config_file_path = sys.argv[1]
with open(config_file_path, 'r') as f:
    config = json.load(f)

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_8bit=True)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")


if config['use_gpu']:
    device = 'cuda'
    # model.to(device)
else:
    device = 'cpu'
    model.to(device)

# wandb.init(project=config['wandb_project'], config=config)

model = PeftModel.from_pretrained(model, config['checkpoint_path'])
# print(dir(model.from_pretrained))


test_file = config['test_file']

model.eval()
total_loss = 0

results = {}
count = 0

class TKVQADataset(Dataset):
    def __init__(self, data, processor):
        super().__init__()
        self.raw_data = data
        self.processor = processor
        self.data = []

        for key, value in self.raw_data.items():
            value['q_id'] = key
            self.data.append(value)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        
        # get corresponding knowledge
        if oracle_flag:
            knowId = item['img_id'].split('_')[0]
            knowledge = kb[knowId]
        else:
            if not movie_flag:
                realImgId = item['img_id']
            else:
                realImgId = item['img_id'].replace('imdb', 'imdb_')

            if realImgId in non_oracle_map:                    
                knowId = non_oracle_map[realImgId][0]['know_id']
                if movie_flag:
                    knowledge = kb[knowId.replace('imdb_', 'imdb')]
                else:
                    knowledge = kb[knowId]
            else:
                knowledge = ''

        # added a shortcut in the next line change it later #TODO
        image = Image.open(item['image_path'])
        question = item['question']

        prompt_with_answer = f"<image>\nUSER: {question}\nContext: {knowledge}\nASSISTANT: </s>"
        inputs = self.processor(text=prompt_with_answer, images=image, return_tensors="pt", padding='max_length', max_length=800, truncation=True) #ideally 220, changed for movie test.

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)


        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }

        return inputs, item['q_id']
  
# Read data from files
def read_data(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

test_data = read_data(test_file)
kb = read_data(config['kb_file'])
oracle_flag = config['use_oracle']
movie_flag = config['movie_flag']
non_oracle_map = read_data(config['non_oracle_map'])

test_dataset = TKVQADataset(test_data, processor)
test_dataloader = DataLoader(test_dataset, batch_size=config['valid_batch_size'], shuffle=False)

pbar = tqdm(test_dataloader, desc=f"Test Loss: 0.0000", dynamic_ncols=True)
results = {}
for batch_idx, batch in enumerate(pbar):

    inputs, q_ids = batch

    inputs = {k:v.cuda() for k,v in inputs.items()}
    generate_ids = model.generate(**inputs, max_new_tokens=100) # 1024 for movie # 400 for scene # 600 for book
    text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    for idx, q_id in enumerate(q_ids):
        results[q_id] = text[idx]

    if batch_idx % 50 == 0:
        with open(config['results_save_path'], "w") as f:
            json.dump(results, f, indent=4)

with open(config['results_save_path'], "w") as f:
    json.dump(results, f, indent=4)
    