import torch
import matplotlib.pyplot as plt
from generator import Generator
from tokenizer import Tokenizer
from autoregressivewrapper import AutoregressiveWrapper
from languagemodel import LanguageModel
from trainer import Trainer



def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    # Tokenize the training data
    tokenized_training_data = tokenizer.tokenize(training_data)
    for _ in range(max_sequence_length):
        # Prepend padding tokens
        tokenized_training_data.insert(0, tokenizer.character_to_token('<pad>'))
    return tokenized_training_data
  
def create_training_sequences(max_sequence_length, tokenized_training_data):
    # Create sequences of length max_sequence_length + 1
    # The last token of each sequence is the target token
    sequences = []
    for i in range(0, len(tokenized_training_data) - max_sequence_length - 1):
        sequences.append(tokenized_training_data[i: i + max_sequence_length + 1])
    return sequences

class Runner(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def run(self, epoch, batch_size):
        # Create the tokenizer
        tokenizer = Tokenizer()

        embedding_dimension = 256
        max_sequence_length = 20
        number_of_tokens = tokenizer.size()

        # Create the model
        model = AutoregressiveWrapper(
          device=self.device,
          gpt_model= LanguageModel(
            device=self.device,
            embedding_dimension=embedding_dimension,
            number_of_tokens=number_of_tokens,
            number_of_heads=4,
            number_of_layers=3,
            dropout_rate=0.1,
            max_sequence_length=max_sequence_length
            ).to(self.device)
        ).to(self.device)

        # Create the training data
        training_data = '. '.join([
            "dogs are friendly animals",
            "many people love their pet dogs",
            "dogs enjoy playing fetch",
            "training dogs requires patience",
            "some dogs are excellent swimmers",
            "dogs provide companionship to humans",
            "search dogs help find lost items",
            "dogs have a strong sense of smell",
            "dogs come in various sizes",
            "dogs are known for their loyalty",
            "dogs can be trained to do tricks",
            "dogs need regular exercise",
            "guide dogs assist the visually impaired",
            "therapy dogs offer comfort in hospitals",
            "dogs have unique personalities",
            "dogs like to dig holes in the yard",
            "police dogs assist law enforcement",
            "dogs communicate through barking",
            "dogs enjoy belly rubs",
            "dogs come in many different breeds",
            "dogs can sense their owners emotions",
            "dogs are known for their wagging tails",
            "many families adopt rescue dogs",
            "dogs are often used in movies",
            "dogs provide security to homes",
            "dogs are great playmates for children",
            "dogs are often part of family photos",
            "some dogs are natural herders",
            "dogs have been human companions for centuries",
            "dogs have a strong pack mentality",
            "dogs have a natural instinct to protect",
            "dogs have excellent night vision",
            "dogs love to chase after squirrels",
            "dogs are used in search and rescue missions",
            "dogs are known for their acute hearing",
            "dogs enjoy sunbathing in the yard",
            "dogs are social animals",
            "dogs are often seen in parks",
            "dogs are curious about their surroundings",
            "dogs have a playful nature",
            "dogs have been bred for specific purposes",
            "dogs are often featured in advertisements",
            "dogs have a special bond with their owners",
            "dogs can be trained to detect drugs",
            "dogs are used as therapy animals for ptsd",
            "dogs love to go for car rides",
            "dogs are expert beggars at the dinner table",
            "dogs provide warmth on cold nights",
            "dogs like to chew on toys",
            "dogs enjoy playing in the snow",
            "dogs have a strong sense of territory",
            "dogs are natural hunters",
            "dogs have a sense of curiosity",
            "dogs are often spoiled with treats",
            "dogs make great hiking companions",
            "dogs can be protective of their toys",
            "dogs are known for their wet noses",
            "dogs are expert nappers",
            "dogs have a keen sense of direction",
            "dogs can be trained to be service animals",
            "dogs love to roll in the grass",
            "dogs are great listeners",
            "dogs have a calming presence",
            "dogs have a preference for certain foods",
            "dogs are always up for a game of fetch",
            "dogs enjoy being part of family gatherings",
            "dogs are often featured in viral videos",
            "dogs are known for their happy faces",
            "dogs are great at reducing stress",
            "dogs are often seen in neighborhood walks",
            "dogs are expert at finding hidden treats",
            "dogs have a natural love for water",
            "dogs provide emotional support",
            "dogs have a talent for escaping yards",
            "dogs enjoy digging up buried treasures",
            "dogs are experts at receiving belly rubs",
            "dogs are known to howl at sirens",
            "dogs have a natural instinct to chase moving objects",
            "dogs enjoy cuddling on the couch",
            "dogs are known for their playfulness",
            "dogs have a special connection with children",
            "dogs love exploring new places",
            "dogs have an incredible sense of smell",
            "dogs enjoy playing with other dogs",
            "dogs have a strong desire to please their owners",
            "dogs like to greet people with tail wags",
            "dogs are often seen in dog parks",
            "dogs have a way of stealing hearts",
            "dogs are known for their unconditional love",
            "dogs have a knack for finding crumbs",
            "dogs are excellent swimmers",
            "dogs make great hiking companions",
            "dogs enjoy chasing after balls",
            "dogs are known for their friendly demeanor",
            "dogs are often seen in obedience classes",
            "dogs have a natural talent for agility",
            "dogs love to sunbathe on lazy afternoons",
            "dogs have a sense of adventure",
            "dogs are often featured in pet magazines",
            "dogs have a way of making people smile",
            "dogs are known for their contagious joy",
        ])

        tokenized_and_padded_training_data = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)
        sequences = create_training_sequences(max_sequence_length, tokenized_and_padded_training_data)

        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        trainer = Trainer(self.device, model, tokenizer, optimizer)
        loss_per_epoch = trainer.train(sequences, epochs=epoch, batch_size=batch_size)

        # Plot the loss per epoch in log scale
        plt.plot(loss_per_epoch)
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        model.save_checkpoint('./trained_model')

        # Generate text
        max_tokens_to_generate = 400
        generator = Generator(self.device, model, tokenizer)
        generated_text = generator.generate(
            max_tokens_to_generate=max_tokens_to_generate,
            prompt="cats",
            padding_token=tokenizer.character_to_token('<pad>')
        )
        print(generated_text.replace('<pad>', ''))