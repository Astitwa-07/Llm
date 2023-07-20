#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torch transformers


# In[3]:


import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question = item['question']
        answer = item['answer']
        inputs = self.tokenizer.encode_plus(question, answer, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def load_data(self, data_path):
        # Load your dataset from the given data_path
        # The dataset should be a list of dictionaries where each dictionary contains 'question' and 'answer' keys

        # Example dataset:
        return [
            {
                'question': 'What is the capital of France?',
                'answer': 'The capital of France is Paris.'
            },
            {
                'question': 'Who wrote the novel "Pride and Prejudice"?',
                'answer': 'The novel "Pride and Prejudice" was written by Jane Austen.'
            },
            # Add more examples...
        ]

def train_model(dataset, model, tokenizer, num_epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {average_loss}")

    # Save the trained model
    model.save_pretrained("trained_model")

def initialize_model():
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def main():
    tokenizer, model = initialize_model()

    # Provide the path to your training dataset
    data_path = "C:/Users/Hrithik Kapil/Dropbox/My PC (LAPTOP-AKEUPEMO)/Downloads/train.json"
    max_length = 512
    batch_size = 4
    num_epochs = 3
    gradient_accumulation_steps = 8


    dataset = CustomDataset(data_path, tokenizer, max_length)
    train_model(dataset, model, tokenizer, num_epochs, batch_size,gradient_accumulation_steps)

    while True:
        question = input("Enter your question (or 'q' to quit): ")
        if question.lower() == 'q':
            break
        answer = generate_answer(question, tokenizer, model)
        print("Answer:", answer)
        print()

if __name__ == '__main__':
    main()


# In[ ]:


import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question = item['question']
        answer = item['answer']
        inputs = self.tokenizer.encode_plus(question, answer, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def load_data(self, data_path):
        # Load your dataset from the given data_path
        # The dataset should be a list of dictionaries where each dictionary contains 'question' and 'answer' keys

        # Example dataset:
        return [
            {
                'question': 'What is the capital of France?',
                'answer': 'The capital of France is Paris.'
            },
            {
                'question': 'Who wrote the novel "Pride and Prejudice"?',
                'answer': 'The novel "Pride and Prejudice" was written by Jane Austen.'
            },
            # Add more examples...
        ]

def train_model(dataset, model, tokenizer, num_epochs, batch_size, gradient_accumulation_steps):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch
    accumulated_steps = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            loss = loss / gradient_accumulation_steps  # Gradient accumulation

            loss.backward()

            if (steps + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulated_steps += 1

            total_loss += loss.item()
            steps += 1

        average_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1} - Average Loss: {average_loss}")

    # Save the trained model
    model.save_pretrained("trained_model")

def initialize_model():
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def main():
    tokenizer, model = initialize_model()

    # Provide the path to your training dataset
    training_data_path = "C:/Users/Hrithik Kapil/Dropbox/My PC (LAPTOP-AKEUPEMO)/Downloads/train.json"
    max_length = 512
    batch_size = 4
    num_epochs = 3
    gradient_accumulation_steps = 8

    dataset = CustomDataset(training_data_path, tokenizer, max_length)
    train_model(dataset, model, tokenizer, num_epochs, batch_size, gradient_accumulation_steps)

    while True:
        question = input("Enter your question (or 'q' to quit): ")
        if question.lower() == 'q':
            break
        answer = generate_answer(question, tokenizer, model)
        print("Answer:", answer)
        print()

if __name__ == '__main__':
    main()


# In[ ]:




