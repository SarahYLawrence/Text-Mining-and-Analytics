import re

def remove_alphabets(text):
    # Replace alphabets with an empty string
    return re.sub('[a-zA-Z\n\s : . , - _ (  )] ? !', '', text)
for k in range(20,22):
    
        file_path=f"C:\\Users\\reihaneh.maarefdoust\\Desktop\\ERisk\\test-llama3-f\\2023-test-answer-Question{k}.txt"
        
        with open(file_path,'r') as file:
            text=file.read()
        answer=['0','1','2','3','4','5','6']
        answer2=['0.','1.','2.','3.','4.','5.','6.']
        
        list1=["0. NOT AT ALL (0)","1. SLIGHTY (1)","2. SLIGHTY (2)","3. MODERATELY (3)","4. MODERATELY (4)","5. MARKEDLY (5)","6. MARKEDLY (6)", 
               "0. NO DAYS","1. 1-5 DAYS", "2. 6-12 DAYS", "3. 13-15 DAYS", "4. 16-22 DAYS", "5. 23-27 DAYS", "6. EVERY DAY"]
        
        list2=[ "NOT AT ALL (0)","SLIGHTY (1)","SLIGHTY (2)","MODERATELY (3)","MODERATELY (4)","MARKEDLY (5)","MARKEDLY (6)", 
               "NO DAYS","1-5 DAYS", "6-12 DAYS", "13-15 DAYS", "16-22 DAYS", "23-27 DAYS"]
        
        list3=["NOT AT ALL","SLIGHTY","SLIGHTY","MODERATELY","MODERATELY","MARKEDLY","MARKEDLY"]
        
        for i in range(0,len(list1)):
            text=text.replace(list1[i], list1[i][0])
        
        for i in range(0,len(list2)):
            text=text.replace(list2[i], list1[i][0])
        
        for i in range(0,len(list3)):
            text=text.replace(list2[i], list1[i+7][0])
    
        for i in range(0,len(answer2)):
            text=text.replace(answer2[i], answer[i])    
    
        text = remove_alphabets(text)
        
        parts = text.split('#')
        for i in range(0,len(parts)):
            if len(parts[i])==0: parts[i]='7'
            if len(parts[i])>1:parts[i]='7'
            if not(parts[i].isdigit()): parts[i]=7
        
        text='#'.join(parts)
     
        text=text.replace('#', '')
        print(text[0:28])
        print(len(text[0:28]))
          
        file_path=f"C:\\Users\\reihaneh.maarefdoust\\Desktop\\ERisk\\test-llama3-f\\clear-2023-test-answer-Question{k}.txt"
        with open(file_path,'w') as file:
           text=file.write(text[0:28])
            

