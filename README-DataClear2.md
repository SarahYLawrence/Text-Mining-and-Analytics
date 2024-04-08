### README

This script, `Erisk-T3-dataclear2.ipynb`, is designed to perform several data preprocessing tasks related to text data stored in CSV files. Below is an overview of the tasks performed by the script:

1. **Merging Text from CSV Files to Create Text Files**:
   - The script reads text data from CSV files located in specified folders.
   - It merges the text data from each CSV file into a single text file.
   - Text files are saved in separate folders corresponding to the year of the data (e.g., '2022txt', '2023txt').

2. **Creating JSON Files with Merged Text**:
   - After merging text data into text files, the script creates JSON files containing information about the text data.
   - Each JSON object includes the file name and the merged text data.
   - Separate JSON files are created for each year (e.g., 'output_json2022.json', 'output_json2023.json').

3. **Summarizing Text**:
   - The script summarizes the merged text data using the Latent Semantic Analysis (LSA) algorithm.
   - Summaries are generated for each text and appended to the corresponding JSON objects.

4. **Generating Sentence Embeddings**:
   - Using the Sentence-BERT (SBERT) model, the script generates vector representations (embeddings) for the text data.
   - Embeddings are generated for each text and added to the corresponding JSON objects.

5. **Updating JSON Files with Embeddings**:
   - The script updates the JSON files with the generated sentence embeddings.

Before running the script, ensure that the necessary libraries are installed by executing the provided pip install commands. Additionally, replace placeholder paths and file names with the appropriate paths and names for your environment.

For any questions or issues, please refer to the original Colab notebook: [Erisk-T3-dataclear2.ipynb](https://colab.research.google.com/drive/1XoGfCMVl9DKBNUwaMmSkl67-vL9d15ZB).