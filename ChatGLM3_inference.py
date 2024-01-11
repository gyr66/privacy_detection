PROMPT = """
现有以下种类：position, name, movie, organization, company, book, address, scene, mobile, email, game, government, QQ（QQ号）, vx（微信号）。
在给定文本中依次找出所有属于种类的命名实体
输出格式是:<命名实体1>[所属种类];<命名实体2>[所属种类];...<命名实体n>[所属种类]
按照给定的输出格式输出，不要输出任何多余的内容！
参考以下例子：
输入文本：鑫视空间（北京）装饰艺术公司总经理佘达建行信用卡50元人民币/次交行信用卡50元人民币/次首先我们邀请中弘北京像素销售总监龙坤先生致辞。
输出文本：<佘达>[name];<建行>[company];<中弘北京像素>[company];<销售总监>[position];<龙坤>[name]
给定文本："""
input_test = "《别告诉我你懂PPT》《不懂项目管理还敢拼职场》《让营销更性感》的作者李治（Liz），《不懂项目管理，还敢拼职场》及《别告诉我你懂PPT》的作者"

'''

'''

import torch
from transformers import AutoModel, AutoTokenizer
import os
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime
import pdb
import re
import csv


run_name = 'news'

### 
# datestr = '20240106-014151'
# datestr = '20240106-201447'
# datestr = '20240107-001132'
datestr = '20240108-021118'

# datestr = datestr + '/checkpoint-'

lr = 2e-5
max_new_tokens = 1024
temperature = 0     #0.4
top_p = 1
output_dir = f"./output/{run_name}-{datestr}-{lr}/checkpoint-5800"

print('output_dir:',output_dir)

lora_path = output_dir + "/pytorch_model.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 加载分词器和模型
model_path = r'/home/platform/GeWei/Projects/ChatGLM3/models'
tokenizer_path = r'/home/platform/GeWei/Projects/ChatGLM3/models'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, load_in_8bit=False, trust_remote_code=True).to(device)

# pdb.set_trace()
# LoRA 模型配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    target_modules=['query_key_value'],
    r=8, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
# pdb.set_trace()

        
if os.path.exists(lora_path):
    
    print('='*20)
    model.load_state_dict(torch.load(lora_path), strict=False)
    print("lora path is loaded!")

def test():
    # with open('./test/104.txt','r',encoding='utf-8') as f:
    #     input_test= f.readline()
    
    print(input_test)
    inputs = tokenizer(PROMPT+input_test, return_tensors="pt").to(device)
    response = model.generate(input_ids=inputs["input_ids"],max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,do_sample=False)
    response = response[0, inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    print(response)
    
    # print("=="*10)
    # print('\n')
    # inputs = tokenizer(input_test, return_tensors="pt").to(device)
    # response = model.generate(input_ids=inputs["input_ids"],max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,do_sample=False)
    # response = response[0, inputs["input_ids"].shape[-1]:]
    # response = tokenizer.decode(response, skip_special_tokens=True)
    # print(response[5:])

def main():
    # with open('./predict.csv','w',newline='') as csvfile:
    #     writer =  csv.writer(csvfile)
    #     writer.writerow(['ID','Category','Pos_b','Pos_e','Privacy'])
    if True:
        test_path = './test/'
        file_count = len(os.listdir(test_path))
        # pdb.set_trace()
        for index in range(0,file_count):
            row =list()
            row.append(index)
                
            with open('./test/'+str(index)+'.txt','r',encoding='utf-8') as f:
                input_test = f.readline()
                input_test = re.sub('"','',input_test)
                

                inputs = tokenizer(PROMPT+input_test, return_tensors="pt").to(device)
                response = model.generate(input_ids=inputs["input_ids"],max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,do_sample=False)
                response = response[0, inputs["input_ids"].shape[-1]:]
                response = tokenizer.decode(response, skip_special_tokens=True)
                print(index,":",response)
                
                if not os.path.exists('./response/base_8k'):
                    os.makedirs('./response/base_8k')
                with open("./response/base_8k/"+str(index)+'.txt','w',encoding='utf-8') as f_rep:
                    f_rep.write(response)
                
                # if ';' in response:
                #     contents = response.split(';')
                # else:
                #     contents = [response]
                # for content in contents:
                #     pdb.set_trace()
                #     temp_row = row
                #     category,privacy = content[1:-1].split('>[')[1],content[1:-1].split('>[')[0]
                #     result = input_test.find(privacy)
                #     temp_row.extend([category,result,result+len(privacy),privacy])
                #     # pdb.set_trace()
                    
                #     writer.writerow(temp_row)  
                    
                        

if __name__ =='__main__':
    test()
    
    # main()
    


    
    torch.cuda.empty_cache()
