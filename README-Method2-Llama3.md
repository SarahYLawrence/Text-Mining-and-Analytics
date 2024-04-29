
```markdown
# Erisk Task3 llama3 Method2

This repository contains code for generating text using the Hugging Face Transformers library. The model used for text generation is the Meta-Llama-3-8B-Instruct model.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
```

## Usage

1. Ensure that your environment has access to a CUDA-enabled GPU for faster computation. If CUDA is available, the code will automatically use it; otherwise, it will default to CPU.

2. Set the `model_id` variable to the desired pre-trained model identifier.

3. Define a list of questions (`question_list`) for which you want to generate answers.

4. Prepare your data in JSON format. The code expects JSON files with the following structure:
   ```json
   [
       {
           "text": "Your text data here",
           "file_name": "Your file name here"
       },
       ...
   ]
   ```

5. Update the file paths for `output_json2022.json` and `output_json2023.json` to point to your data files.

6. Run the code to generate answers for each question. The generated answers will be saved in separate text files for each question and year.

## Sample Questions and Answers

Sample questions and human answers for the years 2022 and 2023 are provided in the code. You can modify or extend these lists as needed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

