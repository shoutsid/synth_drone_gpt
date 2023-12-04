# OpenAI Fine-Tuning Utility

## Introduction
This utility script, `main.py`, is designed to train and test models using the OpenAI API, specifically focusing on fine-tuning for function calling in a conversational AI context. It automates the process of creating a synthetic training dataset, training the model, and validating its performance. In this example we use GPT-4 to generate synthetic code for a drone controller, but the script can be easily adapted to other use cases.

## Requirements
- Python 3.x
- OpenAI API access
- Required Python packages: `openai`, `tenacity`, `numpy`, `json`, `itertools`, `ast`, `pathlib`

## Installation
Clone the repository to your local machine:

```
git clone https://github.com/shoutsid/synth_drone_gpt
cd synth_drone_gpt
```

Install the necessary Python packages (preferably in a virtual environment):

```
pip install -r requirements.txt
```

## Usage
To use this script, follow these steps:

1. **Create a Synthetic Training Dataset**:
   ```
   python main.py train
   ```
   This command will generate a `training.jsonl` dataset for model training.

2. **Fine-Tune the Model**:
   The fine-tuning process is integrated within the `train` command and will commence automatically after the dataset is created.

3. **Validate the Trained Model**:
   ```
   python main.py test
   ```
   This command tests the fine-tuned model using a set of predefined prompts to evaluate its performance. Please ensure to pass the model id created, visible through the openai platform.

## Contributing
Contributions to this project are welcome! If you have suggestions for improvements or want to contribute code, please feel free to open an issue or a pull request.
