# Performer: A Conversational AI Model

Welcome to the Performer repository! This project implements a conversational AI model based on the Performer architecture, leveraging the Cornell Movie Dialogs dataset for training. The model aims to generate human-like responses in a dialogue format.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Conversational AI**: Generate contextually relevant responses based on user input.
- **Transformer Architecture**: Utilizes a Performer model for efficient attention mechanisms.
- **Dataset**: Trained on the Cornell Movie Dialogs dataset for rich conversational data.
- **Custom Training Pipeline**: Integrated training and inference scripts for easy usage.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/RamAnand76/Performer.git
cd Performer
pip install -r requirements.txt
```

## Usage

### Inference

To generate responses using the trained model, run the following command:

```bash
python inference.py
```

You can modify the `input_text` variable in `inference.py` to test with different input prompts.

### Training

To train the model from scratch using the Cornell Movie Dialogs dataset, run:

```bash
python main.py
```

This will preprocess the dataset, initialize the model, and start the training process.

## File Structure

```plaintext
Performer/
├── cornell_movie_dialogs_corpus/
│   ├── chameleons.pdf
│   ├── movie_characters_metadata.txt
│   ├── movie_conversations.txt
│   ├── movie_lines.txt
│   ├── movie_titles_metadata.txt
│   └── raw_script_urls.txt
├── data/
│   ├── dataset.py             # Dataset class for loading conversations
│   └── process_cornell_data.py # Data processing functions for Cornell dataset
├── model/
│   ├── performer.py           # Performer model definition
│   └── utils.py               # Utility functions
├── training/
│   └── train.py               # Training loop
├── inference.py               # Inference script
├── main.py                    # Main script to train the model
├── Performer/                 # Directory for saving model checkpoints
│   └── model.pth              # Trained model weights
└── requirements.txt           # List of required packages

```

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to reach out with any questions or suggestions! Happy coding!


