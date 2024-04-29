import transformers
import torch
import json

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
 
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)   

    
question_list=[" Have you been deliberately trying to limit the amount of food you eat to influence your shape or weight (whether or not you have succeeded)? "
,"Have you gone for long periods of time (8 waking hours or more) without eating anything at all in order to influence your shape or weight?"
,"Have you tried to exclude from your diet any foods that you like in order to influence your shape or weight (whether or not you have succeeded)?"
,"Have you tried to follow definite rules regarding your eating (for example, a calorie limit) in order to influence your shape or weight (whether or not you have succeeded)?"
,"Have you had a definite desire to have an empty stomach with the aim of influencing your shape or weight?"
,"Have you had a definite desire to have a totally flat stomach?"
,"Has thinking about food, eating or calories made it very difficult to concentrate on things you are interested in (for example, working, following a conversation, or reading)?"
,"Has thinking about shape or weight made it very difficult to concentrate on things you are interested in (for example, working, following a conversation, or reading)?"
,"Have you had a definite fear of losing control over eating?"
,"Have you had a definite fear that you might gain weight?"
,"Have you felt fat?"
,"Have you had a strong desire to lose weight?"
,"Over the past 28 days, on how many days have you eaten in secret (ie, furtively)? ... Do not count episodes of binge eating."
,"On what proportion of the times that you have eaten have you felt guilty (felt that you’ve done wrong) because of its effect on your shape or weight? ... Do not count episodes of binge eating."
,"Over the past 28 days, how concerned have you been about other people seeing you eat? ... Do not count episodes of binge eating "
,"Has your weight influenced how you think about (judge) yourself as a person? "
,"Has your shape influenced how you think about (judge) yourself as a person?"
,"How much would it have upset you if you had been asked to weigh yourself once a week (no more, or less, often) for the next four weeks? "
,"How dissatisfied have you been with your weight? "
,"How dissatisfied have you been with your shape?"
,"How uncomfortable have you felt seeing your body (for example, seeing your shape in the mirror, in a shop window reflection, while undressing or taking a bath or shower)? "
,"How uncomfortable have you felt about others seeing your shape or figure (for example, in communal changing rooms, when swimming, or wearing tight clothes)? "]
Human_answers2022 = [
    [6, 5, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 1, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 1],
    [1, 6, 0, 0, 2, 1, 3, 1, 0, 0, 0, 2, 2, 1, 1, 2, 3, 0, 2, 3, 4, 2],
    [2, 1, 4, 6, 1, 6, 4, 5, 4, 6, 6, 6, 4, 6, 2, 2, 6, 1, 3, 6, 5, 4],
    [6, 0, 3, 6, 5, 4, 3, 1, 3, 0, 1, 0, 5, 2, 1, 0, 1, 0, 4, 5, 2, 3],
    [6, 3, 0, 5, 2, 2, 2, 2, 1, 4, 5, 5, 2, 2, 1, 5, 4, 6, 6, 5, 5, 4],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 1, 0, 0, 1, 1, 1],
    [5, 2, 3, 6, 1, 1, 6, 0, 6, 6, 0, 3, 0, 0, 0, 5, 5, 0, 0, 2, 4, 0],
    [6, 6, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6, 2, 2, 2, 6, 6, 4, 6, 6, 6, 6],
    [4, 3, 3, 6, 4, 6, 6, 6, 6, 6, 3, 4, 4, 5, 5, 6, 6, 6, 6, 5, 4, 5],
    [2, 1, 1, 1, 1, 2, 0, 2, 1, 1, 6, 2, 0, 1, 0, 3, 3, 1, 5, 4, 4, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [5, 2, 1, 2, 5, 6, 2, 1, 4, 6, 3, 3, 4, 6, 1, 5, 5, 1, 2, 3, 4, 5],
    [6, 2, 6, 6, 6, 6, 2, 2, 4, 6, 4, 6, 1, 3, 2, 6, 6, 5, 6, 6, 4, 6],
    [6, 1, 6, 3, 6, 6, 6, 5, 6, 6, 6, 6, 3, 4, 2, 5, 5, 6, 6, 5, 6, 5],
    [6, 5, 5, 6, 5, 6, 6, 6, 6, 6, 6, 6, 0, 6, 4, 6, 6, 5, 6, 6, 6, 6],
    [5, 0, 4, 6, 4, 6, 6, 6, 6, 6, 6, 6, 1, 6, 2, 6, 6, 3, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 2, 3, 2, 0],
    [1, 1, 0, 0, 3, 1, 0, 1, 0, 1, 2, 2, 1, 0, 0, 3, 3, 0, 4, 4, 1, 1],
    [6, 3, 6, 6, 6, 4, 6, 6, 6, 6, 6, 3, 1, 0, 0, 5, 2, 1, 2, 1, 4, 6],
    [6, 5, 4, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 1, 5, 5, 6, 6],
    [3, 1, 6, 2, 0, 3, 4, 4, 6, 6, 6, 6, 1, 2, 3, 5, 6, 4, 6, 6, 6, 2],
    [6, 0, 6, 6, 6, 6, 3, 0, 4, 1, 5, 6, 4, 3, 1, 3, 5, 2, 0, 6, 6, 5],
    [3, 3, 3, 0, 2, 6, 3, 3, 5, 5, 5, 4, 1, 1, 1, 5, 5, 6, 5, 5, 5, 6],
    [6, 5, 4, 6, 6, 6, 3, 3, 5, 6, 6, 6, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 1, 6, 6, 6, 2, 6, 5, 6, 6],
    [1, 0, 0, 0, 0, 3, 0, 0, 0, 1, 3, 1, 0, 0, 1, 3, 4, 0, 4, 4, 4, 4],
    [2, 0, 1, 1, 3, 5, 3, 1, 2, 6, 6, 3, 3, 5, 0, 4, 2, 6, 4, 1, 2, 3]
]

Human_answers2023 = [
    [2, 0, 0, 0, 0, 3, 1, 0, 6, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 1, 1, 1],
    [5, 1, 5, 3, 4, 6, 5, 5, 4, 6, 6, 5, 3, 5, 6, 6, 6, 6, 6, 6, 6, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0],
    [0, 6, 6, 0, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 1, 0, 2],
    [1, 0, 1, 0, 0, 2, 0, 0, 0, 0, 4, 2, 0, 1, 0, 1, 2, 4, 2, 2, 2, 2],
    [6, 5, 6, 6, 0, 0, 0, 0, 6, 6, 6, 6, 2, 1, 3, 1, 1, 0, 5, 5, 5, 4],
    [6, 5, 5, 5, 6, 6, 4, 4, 6, 6, 4, 4, 1, 1, 1, 6, 3, 5, 5, 3, 4, 4],
    [5, 4, 6, 4, 5, 6, 1, 1, 1, 4, 3, 6, 0, 3, 0, 3, 3, 3, 4, 3, 2, 3],
    [3, 2, 3, 0, 2, 6, 0, 0, 0, 0, 1, 3, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0],
    [2, 0, 2, 1, 3, 6, 3, 3, 1, 2, 2, 2, 0, 1, 0, 2, 2, 6, 2, 1, 2, 1],
    [5, 0, 4, 6, 4, 6, 4, 3, 2, 4, 4, 6, 1, 3, 2, 5, 5, 6, 5, 5, 5, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 2, 3, 0, 1],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 6, 0, 0, 1, 6],
    [5, 3, 6, 5, 3, 6, 4, 2, 5, 6, 5, 6, 1, 3, 2, 4, 4, 1, 6, 5, 4, 5],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [5, 1, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 3, 3, 6, 5, 5, 4, 5, 5, 6, 5],
    [6, 1, 4, 6, 2, 1, 2, 1, 1, 1, 1, 3, 0, 1, 1, 2, 2, 6, 3, 2, 1, 0],
    [6, 0, 4, 0, 4, 6, 0, 0, 0, 3, 4, 5, 0, 1, 0, 3, 3, 0, 4, 4, 4, 2],
    [4, 1, 1, 0, 4, 1, 0, 0, 2, 2, 4, 3, 0, 1, 0, 0, 6, 1, 4, 6, 5, 6],
    [6, 0, 1, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 0, 2, 6, 4, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2],
    [6, 1, 3, 2, 3, 6, 2, 1, 0, 6, 4, 6, 3, 2, 4, 3, 4, 2, 3, 4, 2, 2],
    [0, 0, 6, 0, 0, 6, 0, 0, 0, 6, 3, 3, 0, 1, 0, 2, 2, 0, 4, 3, 2, 5],
    [6, 5, 6, 6, 6, 6, 5, 5, 6, 6, 6, 6, 5, 3, 3, 5, 5, 0, 6, 4, 3, 6],
    [3, 1, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 4, 0, 1, 1, 0, 0],
    [6, 6, 6, 6, 1, 6, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 2, 1],
    [4, 3, 2, 4, 6, 4, 5, 6, 6, 4, 0, 3, 1, 3, 3, 3, 5, 6, 4, 3, 4, 5],
    [5, 5, 5, 1, 6, 6, 6, 6, 2, 1, 6, 6, 1, 6, 3, 6, 6, 4, 6, 6, 5, 6],
    [1, 0, 4, 1, 0, 6, 1, 1, 3, 1, 1, 1, 0, 1, 0, 4, 4, 1, 2, 4, 4, 4],
    [6, 1, 0, 5, 0, 6, 1, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 3, 6, 6, 2, 6],
    [6, 1, 5, 6, 1, 6, 1, 0, 1, 6, 5, 5, 5, 4, 2, 4, 5, 0, 3, 2, 3, 5],
    [4, 1, 4, 4, 3, 6, 3, 6, 0, 6, 6, 6, 1, 4, 5, 6, 6, 6, 6, 6, 6, 6],
    [6, 1, 2, 6, 2, 6, 6, 6, 6, 6, 6, 6, 2, 6, 0, 6, 6, 0, 6, 6, 6, 6],
    [6, 0, 6, 6, 6, 6, 4, 6, 6, 6, 6, 5, 3, 4, 3, 6, 6, 0, 6, 6, 5, 6],
    [1, 1, 4, 0, 6, 6, 3, 2, 6, 6, 5, 6, 1, 4, 2, 2, 3, 4, 1, 3, 4, 6],
    [0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 3, 2, 4],
    [0, 0, 0, 0, 0, 0, 2, 0, 4, 1, 1, 0, 0, 0, 0, 1, 1, 2, 0, 2, 3, 4],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 6, 2, 0, 1, 0, 6, 6, 0, 6, 6, 6, 0],
    [4, 3, 4, 4, 4, 4, 2, 2, 3, 6, 6, 6, 1, 6, 2, 6, 6, 6, 6, 6, 6, 4],
    [5, 1, 6, 5, 5, 6, 5, 4, 6, 6, 5, 6, 0, 1, 0, 6, 5, 5, 4, 5, 5, 6],
    [6, 4, 0, 0, 2, 6, 0, 0, 3, 1, 2, 2, 1, 2, 0, 4, 4, 6, 4, 4, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2],
    [6, 0, 6, 6, 0, 4, 0, 1, 6, 3, 6, 6, 5, 0, 3, 6, 6, 0, 6, 6, 3, 6],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 3, 0, 0, 0],
    [3, 1, 2, 3, 3, 1, 3, 2, 4, 6, 6, 6, 3, 2, 3, 6, 5, 0, 6, 5, 5, 3],
    [5, 1, 5, 6, 6, 1, 3, 3, 0, 1, 0, 6, 0, 1, 0, 1, 5, 0, 3, 6, 3, 4]
]
# You can access each sublist using data[i], where i is the index of the sublist.



with open("output_json2023.json","r") as file:
    data = json.load(file)
    
post_list=[]
file_name_list=[]
for text_data in data:
        post_list.append(text_data["text"][:])
        file_name_list.append(text_data["file_name"])

for j in range(15,22):    
        answer_list=[]
        answer_list2=[]
        
        for i in range(0,len(post_list)):
            txt='\n'.join(post_list[i])
            
            if len(txt)>5000: 
                text=txt[0:5000]
            else:
                text=txt

    
            instruction = "Answer Multi choice question,do not explain why chose the answer, Do not generate long tesxt,do not use complex words, you must choese one of these are :{0. NOT AT ALL (0)   1. SLIGHTY (1)   2. SLIGHTY (2)   3. MODERATELY (3)   4. MODERATELY (4)   5. MARKEDLY (5)   6. MARKEDLY (6)}, Short answer, do not describe your answer, do not write the question in answer generate"
            messages = [ {"role": "system", "content": f"You are a person who wrote these posts: {text}"},
            {"role": "user", "content": f"{question_list[j]}"},]

            prompt = pipeline.tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)

            terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            outputs = pipeline( prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)    
            find=False
            for k in range(1,20):
                    str1=f"""eRisk2022-T3_Subject{k}.csv"""
                    if (file_name_list[j]==str1):
                            find=True
                            break
            if find==False: Human_answer=7 
            else:            Human_answer=Human_answers2023[k-1][j+1]
            messages1 = [ {"role": "system", "content": f"Human answerd to this question {Human_answer}"},
            {"role": "user", "content": f"{question_list[j]}{instruction}"},]
            prompt = pipeline.tokenizer.apply_chat_template( messages1, tokenize=False, add_generation_prompt=True)
            terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            outputs2 = pipeline( prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
            answer_list2.append(outputs2[0]["generated_text"][len(prompt):])
        combined_answers2 = "".join(answer_list2)  
        file_path = f"2023-Question{j}.txt"
        with open(file_path, 'w') as txtfile:
            txtfile.write(combined_answers2) 

    