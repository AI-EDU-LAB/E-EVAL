import os
import re
from tqdm import tqdm
import torch
import random
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

name_en2zh = {'Middle_School_Chemistry': '初中化学', 'Middle_School_History': '初中历史', 'Middle_School_Geography': '初中地理', 'Middle_School_Politics': '初中政治', 'Middle_School_Mathematics': '初中数学', 'Middle_School_Physics': '初中物理', 'Middle_School_Biology': '初中生物', 'Middle_School_English': '初中英语', 'Middle_School_Chinese': '初中语文', 'Primary_School_Mathematics': '小学小学数学', 'Primary_School_Science': '小学小学科学', 'Primary_School_English': '小学小学英语', 'Primary_School_Chinese': '小学小学语文', 'Primary_School_Ethics': '小学道德与法治', 'High_School_Chemistry': '高中化学', 'High_School_History': '高中历史', 'High_School_Geography': '高中地理', 'High_School_Politics': '高中政治', 'High_School_Mathematics': '高中数学', 'High_School_Physics': '高中物理', 'High_School_Biology': '高中生物', 'High_School_English': '高中英语', 'High_School_Chinese': '高中语文'}

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class Qwen_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name, device):
        super(Qwen_Evaluator, self).__init__(choices, model_name, k)
        # try adding 'mirror="tuna"' and 'resume_download=True' if facing the 'read timed out' problem
        # or directly clone the model
        model_dir = snapshot_download('qwen/Qwen-72B-Chat')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="sequential", trust_remote_code=True).eval()
    
    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, cot=False, save_result_dir=None):
        correct_num = 0
        if save_result_dir:
            if few_shot:
                result = []
            score = []
        # if few_shot:
        #     history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        # else:
        #     history = []
        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot, add_prompt=f"以下是中国关于{name_en2zh[subject_name]}考试的单项选择题，请选出其中的正确答案。\n")
            question = question[0]['value']
            if few_shot:
                history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
                # # try:
                # print(history)
                response, _ = self.model.chat(self.tokenizer, question, do_sample=False, history=history)
                # except Exception as e:
                #     # 捕获异常并处理
                #     print(f"An error occurred while loading SentencePiece model: {e}")
                #     response = random.choice(["A", "B", "C", "D"])
                response = response.strip()
                # For ChatGLM, we use answer extraction in answer-only mode too.
                ans = self.extract_choice(response)
                # print(f"\n{question}{ans}\n--------------------------------------------------------\n{response}")
            else:   # zero-shot by extracting answer from distribution
                history = []
                ans, _ = self.model.chat(self.tokenizer, question, history=history)
                ans = self.extract_choice(ans)
            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            if save_result_dir:
                if few_shot:
                    result.append(response)
                score.append(correct)
        correct_ratio = 100*correct_num/len(answers)
        
        if save_result_dir:
            if few_shot:
                test_df['model_output'] = result
            test_df['correctness'] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

        return correct_ratio
    
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        message = []
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        message.extend(self.format_example(dev_df.iloc[0, :], cot=cot))
        for i in range(1, k):
            message.extend(self.format_example(dev_df.iloc[i, :], cot=cot))
        return message
        
    def format_example(self, line, include_answer=True, cot=False, add_prompt=''):
        content = add_prompt + str(line['question'])
        # print(example)
        for choice in self.choices:
            content += f'\n{choice}. {line[f"{choice}"]}'
        content += '\n答案：'
        if include_answer:
            if cot:
                ans = "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。"
            else:
                ans = line["answer"]
            return [
                {"from": "user", "value": content}, 
                {"from": "assistant", "value": ans}
            ]
        return [{"from": "user", "value": content}]
    
    def extract_cot_answer(self, line, gen_ans):
        m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True
        answer_patterns = [
            r'([ABCD])是正确的',
            r'选项([ABCD])正确',
            r'答案为([ABCD])',
            r'答案是([ABCD])',
            r'答案([ABCD])',
            r'选择([ABCD])',
            r'答案：([ABCD])',
            r'选择答案([ABCD])'
        ]
        # RE extraction
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r'[ABCD]', gen_ans, re.M)
        if len(m) == 1:
            answer = m[0]
            return answer, False
        answer_word_counter = 0
        # only containing one choice-context
        for c in self.choices:
            if str(line[f'{c}']) in gen_ans:
                answer = c
                answer_word_counter += 1
        if answer_word_counter == 1:
            return answer, False
        return '-', False
    
    def generate_dist(self, model, tokenizer, query, history, num_beams=1, max_length=2048,
                      do_sample=False, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"num_beams": num_beams, "do_sample": do_sample, "top_p": top_p, "max_length": 2048,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, **gen_kwargs)
        
        score = outputs.scores[0][0].tolist()
        choice_score = [score[167], score[333], score[251], score[416]]
        ranked_index = [index for index, value in sorted(list(enumerate(choice_score)), key=lambda x:x[1], reverse=True)]
        return self.choices[ranked_index[0]]
    
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