caption = """
Describe each given image individually in detail.
"""
qg_caption = """
Given the multi-image question, generate a caption highlighting the key information related to the question for each image individually in detail, without providing an answer to the question itself. Exclude any options or answers in the caption.\nHere is an attempt:
"""
ccot = """
For the provided image and its associated question, generate a scene graph for each image individually in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question

MUST Exclude any options or answers in the JSON response.
"""
qgcot = """
Your task is to generate preliminary knowledge that aids in answering a given question. You MUST strictly follow the structured response format provided below and NOT answer the given question in the end. Deviations from the format are not allowed. Follow these steps:

### Step 1: Decompose the Question
Break down the question into necessary sub-questions. Identify all the sub-components or aspects of the main question that need to be addressed to understand and solve the problem.

### Step 2: Caption Key Information
For each sub-question, analyze and caption the image summarizing key visual information relevant to the sub-question. The caption should be concise and directly tied to the sub-question.

### Step 3: Use Captions for Auxiliary Knowledge
Utilize the caption as auxiliary knowledge to provide a short, clear answer to each sub-question. These answers should synthesize the captioned information to address the sub-questions effectively.

### Response Format:
Sub-questions:
1. <Sub-question 1>
2. <Sub-question 2>
...

Sub-answers:
1. <Sub-answer 1> (based on the captioned key information)
2. <Sub-answer 2> (based on the captioned key information)
...

### Important Notes:
- Ensure that every sub-question is addressed with a corresponding sub-answer.
- All answers must reference and build upon the captions generated in Step 2.
"""
ddcot = """
You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in choosing the right answer to the question. Note that insufficient information to answer questions is common. The final answer should be one of the options.

Given the context, questions and options, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions based on context, questions and options. Then with the aim of helping humans answer the original question, try to answer the sub-questions. The expected answering form is as follows:\nSub-questions:\n 1. <sub-question 1>\n2. <sub-question 2>\n...\nSub-answers:\n1. <sub-answer 1> or 'Uncertain'\n2. <sub-answer 2> or 'Uncertain'\n...\nAnswer: <One of the options> or 'Uncertain'\n\nFor a question, assume that you do not have any information about the picture, but try to answer the sub-questions and prioritize whether your general knowledge can answer it, and then consider whether the context can help. If sub-questions can be answered, then answer in as short a sentence as possible. If sub-questions cannot be determined without information in images, please formulate corresponding sub-answer into \"Uncertain\". \nOnly use \"Uncertain\" as an answer if it appears in the sub-answers. All answers are expected as concise as possible. \nHere is an attempt:
"""
cocot = """
Describe only the similarities and differences of these images, without providing an answer to the question itself. MUST Exclude any options or answers in the response.
"""
