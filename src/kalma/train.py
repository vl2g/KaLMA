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
from peft import LoraConfig, get_peft_model, TaskType

# set seed
torch.manual_seed(42)

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# save the model in current directory
# model.save_pretrained("./llava-1.5-7b-hf")

config_file_path = sys.argv[1]
with open(config_file_path, 'r') as f:
    config = json.load(f)

if config['use_gpu']:
    device = 'cuda'
    model.to(device)
else:
    device = 'cpu'
    model.to(device)

wandb.init(project=config['wandb_project'], config=config)

lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q_proj", "v_proj"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.CAUSAL_LM,
)

# model = prepare_model_for_int8_training(model, lora_config)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


train_file = config['train_file']
val_file = config['valid_file']
test_file = config['test_file']

# Read data from files
def read_data(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data

train_data = read_data(train_file)
val_data = read_data(val_file)
test_data = read_data(test_file)
kb = read_data(config['kb_file'])

class TKVQADataset(Dataset):
    def __init__(self, data, processor):
        super().__init__()
        self.raw_data = data
        self.processor = processor
        self.data = []

        for key, value in self.raw_data.items():
            self.data.append(value)        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # added a shortcut in the next line change it later #TODO
        image = Image.open(item['image_path'].replace('/data1/abhiram/webqa/VL-T5/','/workspace/VL-T5/'))
        question = item['question']

        # get corresponding knowledge
        knowId = item['img_id'].split('_')[0]
        knowledge = kb[knowId]

        prompt_with_answer = f"<image>\nUSER: {question}\nContext: {knowledge}\nASSISTANT: </s> {str(item['answer'])} [AND] the supporting fact is {str(item['supporting_fact'])}"
        inputs = self.processor(text=prompt_with_answer, images=image, return_tensors="pt", padding='max_length', max_length=400, truncation=True)
        labels = self.processor(text=prompt_with_answer, return_tensors="pt", padding='max_length', max_length=400, truncation=True)['input_ids']

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        labels = labels.squeeze(0)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }

        return inputs, labels
    
train_dataset = TKVQADataset(train_data, processor)
val_dataset = TKVQADataset(val_data, processor)

train_dataloader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['valid_batch_size'], shuffle=False)

num_epochs = config['num_epochs']


optimizer = AdamW(model.parameters(), lr=config['learning_rate'], eps=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * num_epochs)

epoch = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):

    total_loss = 0.0
    
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}, Train Loss: 0.0000", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        start_time = time.time()

        inputs, labels = batch

        if not config['use_accelerate']:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            labels = labels.cuda()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        if int(config['gradient_accumulation_steps']) > 1:
            loss = loss / int(config['gradient_accumulation_steps'])

        optimizer.zero_grad()        
        loss.backward()
        
        if (((batch_idx + 1) % int(config['gradient_accumulation_steps']) == 0) or (batch_idx + 1 == len(train_dataloader))):
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()

        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, batch time: {time.time()-start_time:.2f}, Train Loss: {total_loss / (pbar.n + 1):.4f}")


    pbar.close()

    pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}, Val Loss: 0.0000", dynamic_ncols=True)

    total_val_loss = 0.0

    with torch.no_grad():
        for batch in pbar_val:
            start_time = time.time()

            inputs, labels = batch

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            total_val_loss += loss.item()

            pbar_val.set_description(f"Epoch {epoch + 1}/{num_epochs}, batch time: {time.time()-start_time:.2f}, Val Loss: {total_val_loss / (pbar_val.n + 1):.4f}")

    pbar_val.close()

    average_train_loss = total_loss / len(train_dataloader)
    average_val_loss = total_val_loss / len(val_dataloader)

    wandb.log({"epoch": epoch, "train_loss": average_train_loss, "val_loss": average_val_loss})

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        model.save_pretrained(config['path_to_save'])
    
pbar.close()