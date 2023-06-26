# Databricks notebook source
from transformers import pipeline
import torch

# COMMAND ----------

instruct_pipeline = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

# COMMAND ----------

dbutils.widgets.text('question', '', 'User question')

# COMMAND ----------

question = dbutils.widgets.get('question')

# COMMAND ----------

def get_papers(question):
    pass

# COMMAND ----------

paper = get_paper(request)

# COMMAND ----------

llm_query = f"""Read the given text and answer the question based on the text and your prior knowledge. Text and question are going to be enclosed in quotes and marked TEXT and QUESTION respectively.

TEXT:
'{paper}'

QUESTION:
'{question}'
"""

# COMMAND ----------

result = instruct_pipeline(llm_query)['generated_text']

# COMMAND ----------

instruct_pipeline("What is an exon?")

# COMMAND ----------


