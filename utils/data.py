import os
import datasets
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

class SkipparDataLoader:
    def __init__(self, tokenizer, batch_size = 1, context_size=1024):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        print("== Tokenizer vocab size: ",self.tokenizer.vocab_size)

        self.context_size = context_size
        self.batch_size = batch_size
        self.collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    def apply_tokenizer(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.context_size,
        )
        return tokenized_inputs
    
    def apply_collation(self, batch):
        return self.collator(batch)
    
    # Skippar_train_1.jsonl file has 300k rows, a small sample of the previous paramds_train_1.jsonl which had 23M 
    def build_wikistack_dataloader(
            self, 
            file_path=os.getenv("WIKISTACK_URL"), 
            shuffle = True,
            skip_steps = 0
        ):
        dataset = datasets.load_dataset('json', data_files={'train': file_path}, streaming = True)
        
        train_dataset = dataset["train"]
        

        train_dataset = train_dataset.map(
            lambda samples: self.tokenizer(samples["text"], padding=True, truncation=True, max_length=self.context_size),
            batched=True,
            remove_columns=["id", "text", "url", "meta", "title"]
        ).with_format(type="torch")

        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size = 10000, seed = 42)


        if skip_steps > 0:
            train_dataset = train_dataset.skip(skip_steps * self.batch_size)
            print(f"Streaming data after skipping {skip_steps * self.batch_size} training samples.")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            pin_memory=True,
            # multiprocessing_context="spawn",
        )
        
        return train_loader
    
    def build_c4_dataloader(
            self,
            shuffle,
            skip_steps
        ):
        # self.tokenizer.pad_token = self.tokenizer.special_tokens_map["eos_token"]

        dataset = datasets.load_dataset(
            os.path.realpath(os.getenv("C4_URL")), 
            streaming=True
        )
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(
            lambda samples: self.tokenizer(samples["text"], padding=True, truncation=True, max_length=self.context_size),
            batched=True,
            remove_columns=["text", "url", "timestamp"]
        ).with_format(type="torch")

        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size = 10000, seed = 42)

        if skip_steps > 0:
            train_dataset = train_dataset.skip(skip_steps * self.batch_size)
            print(f"Streaming data after skipping {skip_steps * self.batch_size} training samples.")

        # setting shuffle = True will not be right here as torch dataloader would need the entire size of the dataloader.
        return DataLoader(
            train_dataset,
            collate_fn = self.collator,
            batch_size = self.batch_size,
            pin_memory = True
        )
    
    def build_slim_pajama_dataloader(
            self,
            shuffle,
            skip_steps
        ):
        # self.tokenizer.pad_token = self.tokenizer.special_tokens_map["eos_token"]

        dataset = datasets.load_dataset(
            os.path.realpath(os.getenv("SLIM_PAJAMA_URL")), 
            streaming=True
        )
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(
            lambda samples: self.tokenizer(samples["text"], padding=True, truncation=True, max_length=self.context_size),
            batched=True,
            remove_columns=["text", "meta"]
        ).with_format(type="torch")

        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size = 10000, seed = 42)

        if skip_steps > 0:
            train_dataset = train_dataset.skip(skip_steps * self.batch_size)
            print(f"Streaming data after skipping {skip_steps * self.batch_size} training samples.")

        # setting shuffle = True will not be right here as torch dataloader would need the entire size of the dataloader.
        return DataLoader(
            train_dataset,
            collate_fn = self.collator,
            batch_size = self.batch_size,
            pin_memory = True
        )

        


# def build_c4_dataloader(
#         tokenizer, 
#         context_size, 
#         batch_size,
#         skip_steps
#     ):
#     tokenizer.pad_token = tokenizer.special_tokens_map["eos_token"]

#     collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
#     dataset = datasets.load_dataset(os.path.realpath("/scratch/kevinmahesh/mtech-project-files/datasets/c4/en"), streaming=True)
#     dataset = dataset.map(
#         lambda samples: tokenizer(samples["text"], padding=True, truncation=True, max_length=context_size),
#         batched=True,
#         remove_columns=["text", "url", "timestamp"]
#     ).with_format(type="torch")

#     train_dataset = dataset["train"]

#     if skip_steps > 0:
#         train_dataset = train_dataset.skip(skip_steps * batch_size)
#         print(f"Streaming data after skipping {skip_steps * batch_size} training samples.")

#     # return (
#     #     train_dataset,
#     #     {
#     #         "collate_fn": collator,
#     #     }
#     # )

#     return DataLoader(
#         train_dataset,
#         collate_fn = collator,
#         batch_size = batch_size,
#         pin_memory = True
#     )
