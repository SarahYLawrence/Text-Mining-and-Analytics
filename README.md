# Text-Mining-and-Analytics
## How to Get the Data
Step 1: Go to the Erisk website at https://erisk.irlab.org/

Step 2: If you look under one of the tasks there will be a linked text that says "CLEF 2024 Labs Registration site" and "here" You will need to go to both links.
2024 links:
- "CLEF 2024 Labs Registration site": https://clef2024-labs-registration.dei.unipd.it/registrationForm.php
  - once you open the page you will see a registration form. Fill it out with what you need then submit.
-"here": https://erisk.irlab.org/eRisk2024.html
  - once you open this page scroll to the bottom and click "user agreement". Fill out the user agreement form. go back to the "here" page and scroll to the bottom. Send the agreement form to the person it says to. 

Step 3: wait for an email giving you access to the data.
- Email: The email will give you a link, and password. click the link. Type your team name for the user name and use the password is provided. 

## Task 1

#### Task Overview
This Task involves ranking sentences of the TREC files 
according to their relevance to the set of BDI (Beck’s Depression Inventory) depression symptoms. 
A sentence is considered a BDI symptom when it displays information about the user's state concerning the symptom.

#### Baseline Modle Overview
This baseline model dose the following:
Input:
- Beck's Inventory of Depression (most severe options)
- Author created queries
- Combined queries (based on which queries did the best (BID vs author queries))
- User posts
  
Preprocessing:
- Lowercased the text
- Remove stop words
- Tokenized using Natural Language Toolkit (NLTK) 
- Joined the tokens

Model:
- Universal Sentence Encoder -> Edited_Project
- Roberta -> Edited_Project_bert

Similarity:
- Normalized embeddings
- Calculates the dot product (Comparing user posts embeddings with severe BDI options embeddings and using np.dot() to find the similarities between them)
