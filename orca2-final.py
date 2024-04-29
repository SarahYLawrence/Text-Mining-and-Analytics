import torch
import transformers
import json
if torch.cuda.is_available():
    torch.set_default_device("cuda") 

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



with open("output_json2023.json","r") as file:
    data = json.load(file)
    
post_list=[]
file_name_list=[]
for text_data in data:
        post_list.append(text_data["text"][:])
        file_name_list.append(text_data["file_name"])

model = transformers.AutoModelForCausalLM.from_pretrained(
    "microsoft/Orca-2-7b",
    device_map='auto',
    load_in_8bit=True)

tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/Orca-2-7b",
        use_fast=False,
    )

answer_list=[]
user_message=" you are who write the text now answer the question"
for i in range(0,len(post_list)):
    txt='\n'.join(post_list[i])
    if len(txt)>1000: 
            text=txt[0:1000]
    else:
            text=txt

    for j in range(0,15):
        #user_message=
        prompt = f"""<|im_start|>system\n content: {text} Give me an answer for Multiple-Choice Question {question_list[j]}: 0. NO DAYS	1. 1-5 DAYS	2. 6-12 DAYS	3. 13-15 DAYS	4. 16-22 DAYS	5. 23-27 DAYS	6. EVERY DAY <|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"""
        inputs = tokenizer(prompt,return_tensors='pt')
        output_ids = model.generate(inputs["input_ids"],max_length=4096,)
        answer = tokenizer.batch_decode(output_ids)[0]
        find=False
        for k in range(1,48):
                    str1=f"""eRisk2023-T3_Subject{k}.csv"""
                    if (file_name_list[j]==str1):
                            find=True
                            break
            
        if find==False: Human_answer=7 
        else:            Human_answer=Human_answers2023[k][j]
        second_turn_user_message = f"person who written the post answer is {Human_answer}  '0. NO DAYS	1. 1-5 DAYS	2. 6-12 DAYS	3. 13-15 DAYS	4. 16-22 DAYS	5. 23-27 DAYS	6. EVERY DAY',answer base on the key points of your first answer."
        #second_turn_user_message = f""" question_list{[j]} chose answer from Multiple-Choice Questions:0. NO DAYS	1. 1-5 DAYS	2. 6-12 DAYS	3. 13-15 DAYS	4. 16-22 DAYS	5. 23-27 DAYS	6. EVERY DAY"""
        #0.NOT AT ALL (0), 1.SLIGHTY (1), 2.SLIGHTY (2),3.MODERATELY (3), 4.MODERATELY (4), 5.MARKEDLY (5), 6.MARKEDLY (6)"""
        second_turn_message_in_markup = f"\n<|im_start|>user\n{second_turn_user_message}<|im_end|>\n<|im_start|>assistant"
        second_turn_tokens = tokenizer(second_turn_message_in_markup, return_tensors='pt', add_special_tokens=False)
        second_turn_input = torch.cat([output_ids, second_turn_tokens['input_ids'][:400]], dim=1)
        output_ids_2 = model.generate(second_turn_input,)
        second_turn_answer = tokenizer.batch_decode(output_ids_2)[0]
        answer_list.append(second_turn_answer)
        
    file_path = f"{data[i]['file_name']}-orca.txt"
    with open(file_path, 'w') as txtfile:
            txtfile.write(second_turn_answer)


