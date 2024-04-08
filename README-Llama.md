# Llama-2-7b Chatbot

This repository contains a Jupyter Notebook for running the Llama-2-7b chatbot model on Google Colab. Llama-2-7b is a conversational model from Hugging Face's model hub designed to generate responses to questions based on input text.

## Getting Started

To get started with Llama-2-7b Chatbot on Colab, follow these steps:

1. Open the notebook `Erisk-T3-Llama.ipynb` in Google Colab.
2. Run the cells in the notebook to execute the code and generate answers to questions.

## Overview

The notebook demonstrates how to:
- Load the Llama-2-7b chatbot model and tokenizer from Hugging Face.
- Use the chatbot model to generate responses to predefined questions based on input text.
- Save the generated answers to a CSV file.

## Usage

The notebook provides a set of predefined questions and input text extracted from JSON data. It uses the Llama-2-7b model to generate answers to these questions based on the provided input text.

## Output

The generated answers are saved to a CSV file named `answer-Q.csv`. Each row in the CSV file contains a question, its corresponding answer, and the subject of the input text.

## Dependencies

- Python 3.x
- PyTorch
- Transformers
- Hugging Face model hub
- JSON library
- CSV library

## Note

This README provides a brief summary of the contents of the notebook. For detailed instructions and code, refer to the notebook file `Erisk-T3-Llama.ipynb`.

