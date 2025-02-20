import deepspeed
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# A dummy dataset for demonstration purposes.
class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, length=10000, seq_length=128):
        self.len = length
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        # Generate random tokens by sampling from tokenizer vocab indices
        input_ids = torch.randint(low=0, high=self.tokenizer.vocab_size, size=(self.seq_length,))
        # For language modeling, the target is usually the same as input shifted.
        labels = input_ids.clone()
        return input_ids, labels

    def __len__(self):
        return self.len

def main():
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed()

    # Load the pretrained GPT-2 model and tokenizer from Hugging Face.
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # DeepSpeed configuration file.
    ds_config = "ds_config.json"

    # Initialize DeepSpeed engine.
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Create the dataset and dataloader.
    dataset = RandomTextDataset(tokenizer, length=10000, seq_length=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Define the loss function (GPT-2 typically uses CrossEntropyLoss).
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 3
    for epoch in range(num_epochs):
        for i, (input_ids, labels) in enumerate(dataloader):
            # DeepSpeed handles device placement, but you can ensure tensors are on the correct device:
            input_ids = input_ids.to(model_engine.local_rank)
            labels = labels.to(model_engine.local_rank)

            # Forward pass. Note: GPT-2 returns a tuple; the first element is logits.
            outputs = model_engine(input_ids)
            logits = outputs[0]
            
            # Reshape logits and labels for computing the loss.
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            model_engine.backward(loss)
            model_engine.step()

            # Log progress on the master node.
            if model_engine.global_rank == 0 and i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss {loss.item()}")

if __name__ == "__main__":
    main()
