# Databricks notebook source
from transformers import pipeline
import torch

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up

# COMMAND ----------

dbutils.widgets.text('question', '', 'User question')

# COMMAND ----------

question = dbutils.widgets.get('question')

# COMMAND ----------

instruct_pipeline = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Function Definitions

# COMMAND ----------

def get_papers(question):
    pass

# COMMAND ----------

def format_paper(paper):
    return f"{paper['title']}\n{paper['abstract']}\n"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------

papers = get_papers(question)
papers_str = '\n'.join([format_paper(p) for p in papers])

# COMMAND ----------

llm_query_template = """Read the given text and answer the question based on the text and your prior knowledge. Text and question are going to be enclosed in single quotes and marked TEXT and QUESTION respectively.

TEXT:
'{papers_str}'

QUESTION:
'{question}'
"""

# COMMAND ----------

result = instruct_pipeline(llm_query_template.format(papers_str=papers_str, question=question))

# COMMAND ----------

print(resutl['generated_text'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Tests

# COMMAND ----------

test_text = """The kidneys operate at the interface of plasma and urine by clearing molecular waste products while retaining valuable solutes. Genetic studies of paired plasma and urine metabolomes may identify underlying processes. We conducted genome-wide studies of 1,916 plasma and urine metabolites and detected 1,299 significant associations. Associations with 40% of implicated metabolites would have been missed by studying plasma alone. We detected urine-specific findings that provide information about metabolite reabsorption in the kidney, such as aquaporin (AQP)-7-mediated glycerol transport, and different metabolomic footprints of kidney-expressed proteins in plasma and urine that are consistent with their localization and function, including the transporters NaDC3 (SLC13A3) and ASBT (SLC10A2). Shared genetic determinants of 7,073 metabolite–disease combinations represent a resource to better understand metabolic diseases and revealed connections of dipeptidase 1 with circulating digestive enzymes and with hypertension. Extending genetic studies of the metabolome beyond plasma yields unique insights into processes at the interface of body compartments."""

test_question = "How many associations exists between urine and plasma metabolites?"

# COMMAND ----------

llm_query = f"""Read the given text and answer the question based on the text. Text and question are going to be enclosed in single quotes and marked TEXT and QUESTION respectively.

TEXT:
'{test_text}'

QUESTION:
'{test_question}'
"""

# COMMAND ----------

instruct_pipeline(llm_query_template.format(papers_str=test_text, question=test_question))

# COMMAND ----------


