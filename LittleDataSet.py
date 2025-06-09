import torch

from torch.utils.data import Dataset

import json



class dataset_pre(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                itemdata = json.loads(line.strip())
                self.data.append(itemdata)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # input = torch.tensor(input_ids[:-1], dtype=torch.long)
        # label = torch.tensor(input_ids[1:], dtype=torch.long)
        # loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        input = input_ids[:-1]
        label = input_ids[1:]
        loss_mask = loss_mask[1:]
        return input, label, loss_mask


class dataset_sft(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_token = "<|im_start|>assistant"
        self.eos_token = "<|im_end|>"
        self.bos_id = tokenizer(self.bos_token, add_special_tokens=False).input_ids
        self.eos_id = tokenizer(self.eos_token, add_special_tokens=False).input_ids
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                item = line.strip()
                item_data = json.loads(item)
                self.data.append(item_data)

    def __len__(self):
        return len(self.data)

    def _lossmask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_len)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.data[index]

        msg = sample['conversations']

        message = self.tokenizer.apply_chat_template(msg, tokenize=False,
                                                     add_generation_prompt=False)

        encoding = self.tokenizer(
            message,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze(0).tolist()
        loss_mask = self._lossmask(input_ids)

        input = torch.tensor(input_ids[:-1], dtype=torch.long)
        label = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        # input = input_ids[:-1]
        # label = input_ids[1:]
        # loss_mask = loss_mask[1:]
        return input, label, loss_mask


class dataset_dpo(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_token = "<|im_start|>assistant"
        self.eos_token = "<|im_end|>"
        self.bos_id = tokenizer(self.bos_token, add_special_tokens=False).input_ids
        self.eos_id = tokenizer(self.eos_token, add_special_tokens=False).input_ids
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                itemdata = json.loads(line.strip())
                self.data.append(itemdata)

    def __len__(self):
        return len(self.data)

    def _lossmask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_len)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.data[index]
        # message = sample['text'],
        question = sample['question']
        chosen = sample['chosen']
        reject = sample['rejected']

        chosen_prompt = [{"role": "user", "content": question}, {"role": "assistant", "content": chosen}]
        reject_prompt = [{"role": "user", "content": question}, {"role": "assistant", "content": reject}]
        chosen_msg = self.tokenizer.apply_chat_template(chosen_prompt, tokenize=False, add_generation_prompt=False)
        reject_msg = self.tokenizer.apply_chat_template(reject_prompt, tokenize=False, add_generation_prompt=False)
        chosen_encoding = self.tokenizer(
            chosen_msg,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        reject_encoding = self.tokenizer(
            reject_msg,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        #<im_start>user xxxx<im_end><im_start>assistant yyyy<im_end>
        chosen_input_ids = chosen_encoding.input_ids.squeeze(0).tolist()
        chosen_loss_mask = self._lossmask(chosen_input_ids)
        # <im_start>user xxxx<im_end><im_start>assistant yyyy {}
        chosen_input = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        # {} user xxxx<im_end><im_start>assistant yyyy<im_end>
        chosen_label = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        chosen_loss_mask = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        # ------------------------------------------------------------------------
        # <im_start>user xxxx<im_end><im_start>assistant yyyy<im_end>
        reject_input_ids = reject_encoding.input_ids.squeeze(0).tolist()
        reject_loss_mask = self._lossmask(reject_input_ids)
        # <im_start>user xxxx<im_end><im_start>assistant yyyy {}
        reject_input = torch.tensor(reject_input_ids[:-1], dtype=torch.long)
        # {} user xxxx<im_end><im_start>assistant yyyy<im_end>
        reject_label = torch.tensor(reject_input_ids[1:], dtype=torch.long)
        reject_loss_mask = torch.tensor(reject_loss_mask[1:], dtype=torch.long)

        return  chosen_input, chosen_label, chosen_loss_mask, reject_input, reject_label, reject_loss_mask