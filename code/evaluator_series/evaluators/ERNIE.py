import os
import random
from tqdm import tqdm
import openai
from evaluators.evaluator import Evaluator
from time import sleep
import re
import requests
import json

name_en2zh = {'Middle_School_Chemistry': '初中化学', 'Middle_School_History': '初中历史', 'Middle_School_Geography': '初中地理', 'Middle_School_Politics': '初中政治', 'Middle_School_Mathematics': '初中数学', 'Middle_School_Physics': '初中物理', 'Middle_School_Biology': '初中生物', 'Middle_School_English': '初中英语', 'Middle_School_Chinese': '初中语文', 'Primary_School_Mathematics': '小学小学数学', 'Primary_School_Science': '小学小学科学', 'Primary_School_English': '小学小学英语', 'Primary_School_Chinese': '小学小学语文', 'Primary_School_Ethics': '小学道德与法治', 'High_School_Chemistry': '高中化学', 'High_School_History': '高中历史', 'High_School_Geography': '高中地理', 'High_School_Politics': '高中政治', 'High_School_Mathematics': '高中数学', 'High_School_Physics': '高中物理', 'High_School_Biology': '高中生物', 'High_School_English': '高中英语', 'High_School_Chinese': '高中语文'}

class ERNIE_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name):
        super(ERNIE_Evaluator, self).__init__(choices, model_name, k)
        # openai.api_base = "https://api.lqqq.ltd/v1"
        # openai.api_key = api_key

    def get_access_token(self):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """
            
        url = "https://aip.baidubce.com/oauth/2.0/token?client_id=yW94TYcnt2YcK5dT1uEs2ipv&client_secret=XLWu0n8jOsb9NIFo4GG91MIsioGVebYr&grant_type=client_credentials"
        
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    def format_example(self,line,include_answer=True,cot=False):
        example=line['question']
        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        example+='\n答案：'
        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            return [
                {"role":"user","content":example},
            ]
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt=[
            {
                "role":"system",
                "content":f"你是一个中文人工智能助手，以下是中国关于{name_en2zh[subject]}考试的单项选择题，请选出其中的正确答案。"
            }
        ]
        k=self.k
        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            if i==0:
                tmp[0]["content"]=f"以下是中国关于{name_en2zh[subject]}考试的单项选择题，请选出其中的正确答案。\n\n"+tmp[0]["content"]
            prompt+=tmp
        return prompt

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None,cot=False):
        # url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + self.get_access_token()
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + self.get_access_token()  

        headers = {
            'Content-Type': 'application/json'
        }
        
        # response = requests.request("POST", url, headers=headers, data=payload)
        
        # dictionary = json.loads(response.text)
        
        # print(dictionary["result"])
        correct_num = 0
        if save_result_dir:
            result = []
            score=[]
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df,cot=cot)
        else:
            few_shot_prompt=[
                {
                    "role":"system",
                    "content":f"你是一个中文人工智能助手，以下是中国关于{name_en2zh[subject_name]}考试的单项选择题，请选出其中的正确答案。"
                }
            ]
        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"]=f"你是一个中文人工智能助手，以下是中国关于{name_en2zh[subject_name]}考试的单项选择题，请直接给出正确答案对应的选项。\n\n"+full_prompt[-1]["content"]
            response=None
            # payload = json.dumps({
            # "messages": [
            #     {
            #         "role": "user",
            #         "content": "商标著作权登记与商标注册的区别是什么？"
            #     }
            # ]
            # })
            full_prompt = full_prompt[1:]
            payload = json.dumps({
                "messages": full_prompt
            })
            # response = openai.ChatCompletion.create(
            #     model=self.model_name,
            #     messages=full_prompt,
            #     temperature=0.
            # )
            response = requests.request("POST", url, headers=headers, data=payload)
            dictionary = json.loads(response.text)
            # print(dictionary)
            response_str = dictionary["result"]
            if cot:
                ans_list=self.extract_choice(response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"答案为(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"选项(.+?)是正确的。",response_str)

                if len(ans_list)==0:
                    correct=0
                else:
                    if self.exact_match(ans_list[-1],row["answer"]):
                        correct_num+=1
                        correct=1
                    else:
                        correct=0
            else:
                response_str=response_str.strip()
                if few_shot:
                    if len(response_str)>0:
                        response_str = self.extract_choice(response_str)
                        if self.exact_match(response_str,row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        if len(ans_list)>0 and (ans_list[-1]==row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output']=result
            test_df["correctness"]=score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return correct_ratio

    def extract_ans(self,response_str):
        pattern=[
            r"^选([A-D])",
            r"^选项([A-D])",
            r"故选(.+?)。",
            r"故选：(.+?)。",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        ans_list=[]
        if response_str[0] in ["A",'B','C','D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        return ans_list
    
    def extract_choice(self, response):
        response = str(response)
        if response[0] in ['A', 'B', 'C', 'D']:
            return response[0]
        # 1. Single match
        patterns = [
            (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
            (r'答案(是|为)选项 ?([ABCD])', 2),
            (r'故?选择?：? ?([ABCD])',1),
            (r'([ABCD]) ?选?项(是|为)?正确',1),
            (r'正确的?选项(是|为) ?([ABCD])',2),
            (r'答案(应该)?(是|为)([ABCD])',3),
            (r'选项 ?([ABCD]) ?(是|为)?正确',1),
            (r'选择答案 ?([ABCD])',1),
            (r'答案?：?([ABCD])',1),
            (r'([ABCD])(选?项)?是?符合题意',1),
            (r'答案选项：? ?([ABCD])', 1), # chatglm
            (r'答案(选项)?为(.*?)([ABCD])', 3), # chatgpt

        ]
        for pattern,idx in patterns:
            m = re.search(pattern, response, re.M)
            if m:
                answer = m.group(idx)
                assert answer in ['A', 'B', 'C', 'D']
                return answer

        # 2. Recursive match
        patterns = [
            (r'([ABCD])(.*?)当选', 1),
            (r'([ABCD])(.*?)正确', 1),
        ]
        for pattern,idx in patterns:
            m = re.search(pattern, response, re.M)
            if m:
                while m:
                    answer = m.group(idx)
                    m = re.search(pattern, m.group(0)[1:], re.M)
                assert answer in ['A', 'B', 'C', 'D']
                return answer

        # 3. Weak single match
        patterns = [
            (r'[^不]是：? ?([ABCD])', 1),
        ]
        for pattern,idx in patterns:
            m = re.search(pattern, response, re.M)
            if m:
                answer = m.group(idx)
                assert answer in ['A', 'B', 'C', 'D']
                return answer

        # 4. Check the only mentioend choices
        pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
        m = re.match(pattern, response)
        if m:
            answer = m.group(1)
            assert answer in ['A', 'B', 'C', 'D']
            return answer

        return ['A', 'B', 'C', 'D'][random.randint(0,3)]