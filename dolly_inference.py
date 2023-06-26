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

test_text = """If data is the cornerstone to AI and ML, biotech should seemingly be awash in opportunity. The size of datasets is increasing rapidly with the rise of lab automation technologies, the fast pace at which drug discovery unfolds, and the sheer size of data in fields that do screening and sequencing in high-throughput. It’s predicted that biotech data collection will surpass all other fields, hurtling past even astronomy, to have the largest data volume by 2025.

However, the abundance of data in pharma and the life sciences is widely understood to be a blessing and a curse. This is due to the heterogeneous and hierarchical nature of scientific data, which is perpetuated by how the data is generated, organized, and interpreted.

In part, this pitfall occurs because data modeling is simply trickier in biology. In other fields where ML has momentum, such as online commerce, advertising, finance, or media, data models are easier to pin down and agree upon, industry-wide. This lowers the barrier to storing, standardizing, and analyzing across datasets. Meanwhile, scientific data modeling often varies wildly – even across a single team in an organization. For instance, the featurization of a genome can be debated endlessly, as the optimal representation has complex dependencies on the organism, the biological tools for manipulating them, and how uncertainty and updates are handled.

In addition, training ML models in biology usually requires that you have captured a lot of process data and metadata to fuel models. Scientific observation of how a molecule performs is usually a complex function of how the molecule was produced, isolated, tested. For example, pooling data to measure yield or performance of different proteins likely would have terrible generalizability if you neglect any of tens of potential factors like the organism that produced it, or the scale at which the protein was produced. Models that unknowingly mix apples and oranges typically do not generalize to new data well in biotech.

The mistake that we commonly see is that companies jump ahead, and do data science before solving the hard foundational problems of establishing the pipeline and flow of data for analysis. They don’t pause to operationalize the flow of data coming from experimental pipelines for usage by data scientists. As un-sexy as it is, companies need to build their data strategy and systems before benefitting from ML. Beyond simply collecting data, doing so in a standardized way and anticipating how the data should be consumed, is key. Companies need to design their data systems for the analytics and AI and ML they aim to layer on top."""

test_question = "What is the main obstacle to applying ML in life sciences?"

# COMMAND ----------

llm_query = f"""Read the given text and answer the question based on the text and your prior knowledge. Text and question are going to be enclosed in single quotes and marked TEXT and QUESTION respectively.

TEXT:
'{test_text}'

QUESTION:
'{test_question}'
"""

# COMMAND ----------

instruct_pipeline(llm_query_template.format(papers_str=test_text, question=test_question))
