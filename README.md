# Text-Mining-and-Analytics
## Task Overview
This Task involves ranking sentences of the TREC files 
according to their relevance to the set of BDI (Beck’s Depression Inventory) depression symptoms. 
A sentence is considered a BDI symptom when it displays information about the user's state concerning the symptom.

## Baseline Modle Overview
This baseline model dose the following:
Input:
- Beck's Inventory of Depression (most severe options)
- User posts
  
Preprocessing:
- Lowercased the text
- Remove stop words
- Tokenized using Natural Language Toolkit (NLTK) 
- Joined the tokens

Model:
- Universal Sentence Encoder
Similarity:
- Normalized embeddings
- calculates the dot product

## How to Get the Data
Step 1: Go to the Erisk website at https://erisk.irlab.org/  - Note this project used the 2024 data  

Step 2: If you look under task one there will be a linked text that says "CLEF 2024 Labs Registration site" and "here" you will need to go to both links.
2024 links:
- "CLEF 2024 Labs Registration site": https://clef2024-labs-registration.dei.unipd.it/registrationForm.php
  - once you open the page you will see a registration form. Fill it out with what you need then submit. To run this project what should be selected is "eRisk - Early Risk Detection on the Internet" and "task one".
-"here": https://erisk.irlab.org/eRisk2024.html
  - once you open this page scroll to the bottom and click "user agreement". Fill out the user agreement form. go back to the "here" page and scroll to the bottom. Send the agreement form to the person it says to. 

Step 3: wait for an email giving you access to the data.
- Email: The email will give you a link, and password. click the link. Type your team name for the user name and use the password is provided. 

## What to install
- pip install PyPDF2
- pip install nltk
- pip install numpy
- pip install gensim
- pip install tensorflow
- pip install tensorflow tensorflow-hub

## Trouble Shooting
Taking a long time: 
- It does take an hour and a half for the embeddings to complete.
  
It's using a lot of CPU: 
- The multiprocessing that is being performed will take up all available cores except one.
- Use the code under the (# TODO) and delete (Embeding sentences) code If you do not want to use multiprocessing and want a progress bar.

