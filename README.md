![image](https://github.com/AI-EDU-LAB/E-EVAL/assets/56249874/0acf7a53-7854-4b46-bffc-16e2df1639d6)


<p align="center">
   🌐 <a href="https://eevalbenchmark.com/" target="_blank">Website</a> • 🤗 <a href="https://huggingface.co/datasets/E-EVAL/E-EVAL" target="_blank">Hugging Face</a>  • 📃 <a href="https://arxiv.org/abs/2401.15927" target="_blank">Paper</a> 
</p>

## E-EVAL

E-Eval is a Chinese K12 educational assessment benchmark for large language models, covering 4,391 multiple-choice questions in 11 different subjects, divided into three levels of difficulty. More details can be found in our [paper](https://arxiv.org/abs/2401.15927).


## News

## Table of Contents

- Leaderboard
- Data
- How to Evaluate on E-Eval
- How to submit
- Citation
- Acknowledgement
- TODO

## Leaderboard

The following lists the zero-shot and five-shot accuracy of models evaluated in our initial version. Please visit our official [Leaderboard](https://eevalbenchmark.com/leaderboard) for the latest models and their detailed results in each subject. We have noticed that for many models fine-tuned on specific instructions, the zero-shot results are better than the few-shot ones.

#### Zero-shot && five-shot

| Model                        | 0-shot answer-only | 5-shot answer-only | 5-shot COT | Average |
|------------------------------|:-------------------:|:-------------------:|:-----------:|:--------:|
| Qwen-72b                      |               89.0 |               88.7 |       88.8 |    88.8 |
| Ernie-Bot 4.0                 |               86.7 |               85.2 |       84.6 |    85.5 |
| Yi-34b-chat                   |               72.5 |               81.4 |       76.6 |    76.8 |
| Ernie-Bot                     |               76.1 |               75.7 |       75.7 |    75.8 |
| GPT-4                         |               70.5 |               73.8 |       67.4 |    70.6 |
| Yi-6b-chat                    |               68.8 |               71.2 |       66.5 |    68.8 |
| chatglm3-6b                   |               72.9 |               59.3 |       65.0 |    65.7 |
| Qwen-7b                       |               58.7 |               60.4 |       60.4 |    59.9 |
| baichuan2-13b-chat            |               56.1 |               60.9 |       56.1 |    57.7 |
| baichuan2-7b-chat             |               55.2 |               56.2 |       52.9 |    54.8 |
| GPT-3.5                       |               54.5 |               56.9 |       52.3 |    54.6 |
| Chinese-Alpaca-2-13B          |               44.8 |               46.2 |       38.7 |    43.2 | 
| Educhat-base-002-13b          |               37.1 |               40.6 |       36.1 |    37.9 |
| Educhat-sft-002-13b           |               33.2 |               39.4 |       36.1 |    36.2 |
| Chinese-LLaMA-2-13B           |               35.7 |               38.9 |       33.2 |    35.9 | 
| Educhat-sft-002-13b-baichuan  |               54.0 |               14.4 |       38.1 |    35.5 |
| Educhat-base-002-7b           |               30.4 |               27.9 |       29.3 |    29.2 |


## How to Evaluate on E-Eval

Normally you can directly take the model's generations and extract the answer token (i.e. A,B,C,D) from it with simple regular expressions. In few-shot evaluation, the model usually follows the given template thus this is easy. Sometimes, however, especially in zero-shot evaluation for models without experiencing instruction tuning, the model may not follow the instruction well to give a well-formatted generation, in this case we recommend computing the probability of "A", "B", "C", "D" and take the most likely one as the answer -- this is a constrained decoding approach and was used in the official [MMLU test code](https://github.com/hendrycks/test/blob/4450500f923c49f1fb1dd3d99108a0bd9717b660/evaluate.py#L88). Such a probability approach is not applicable for chain-of-thought settings. 

We use the following prompt when evaluating the models in our first release:
#### answer-only prompt
```
以下是中国关于{科目}考试的单项选择题，请选出其中的正确答案。

{题目1}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
答案：A

[k-shot demo, note that k is 0 in the zero-shot case]

{测试题目}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
答案：
```

#### chain-of-thought prompt

```
以下是中国关于{科目}考试的单项选择题，请选出其中的正确答案。

{题目1}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
答案：让我们一步一步思考，
1. {解析过程步骤1}
2. {解析过程步骤2}
3. {解析过程步骤3}
所以答案是A。

[k-shot demo, note that k is 0 in the zero-shot case]

{测试题目}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
答案：让我们一步一步思考，
1. 
```

## How to Submit

You need to first prepare a UTF-8 encoded JSON file with the following format, please refer to [submission_example.json](https://github.com/AI-EDU-LAB/E-EVAL/blob/b720ebdc5e6f5d1b9086962b17cdf7acc74f872f/E-EVAL_sample.json) for details.

  ```
  ## key within each subject is the "id" field from the dataset
  {
      "high_school_biology": {
          "0": "A",
          "1": "B",
          "2": "B",
          ...
      },
      
      "subject_name":{
      "0":"ans_1",
      "1":"ans_2",
      ...
      }
      ....
  }
  ```
  Then you can submit the prepared json file [here](https://eevalbenchmark.com/user_interface), **note that you need to first log in to access the submission page**.
## Citation
```
@article{hou2024eeval,
            title={E-EVAL: A Comprehensive Chinese K-12 Education Evaluation Benchmark for Large Language Models},
            author={Jinchang Hou and Chang Ao and Haihong Wu and Xiangtao Kong and Zhigang Zheng and Daijia Tang and Chengming Li and Xiping Hu and Ruifeng Xu and Shiwen Ni and Min Yang},
            journal={https://arxiv.org/abs/2401.15927},
            year={2024}
}
```
## Acknowledgement
Thanks to [UNION INFORMATION](https://szunion-info.com/) for their support of this work.

## TODO

- [x] add zero-shot results
- [ ] incorporate into openai eval
