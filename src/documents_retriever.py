from pathlib import Path
import re
import requests
import json
import os

def get_ducuments_from_url() -> list[dict]:

    fromdateyear=2018
    numberofdays=100

    documents = []

    for fromdatemonth in range(8,9):

        documents = []

        for fromdateday in range(21,28):
            fromdatedayplus=fromdateday + 1
            #url = 'https://api.biorxiv.org/details/biorxiv/2018-08-21/2018-08-28/45'
            url = (f'https://api.biorxiv.org/details/biorxiv/{fromdateyear}-0{fromdatemonth}'
                + f'-{fromdateday}/{fromdateyear}-0{fromdatemonth}-{fromdatedayplus}/{numberofdays}')
            # breakpoint()
            #print("url: " + url)
            response = requests.get(url)
            
            #print("status code: " + str(r.status_code)) 


            if response.status_code == 200:
                documents.append(response.json())

            else:
                print ("error",response.status_code)
    return documents

def get_documents_locally(dir="docs/") -> list[dict]:
    dir = Path(dir)

    paths = os.listdir(dir)
    documents = []
    for file in paths:
        if not re.match(".*.json", file):
            continue
        with open(dir / file, "r", encoding="utf-8") as f:
            content = json.load(f)
        documents.append(content)
    return documents

def get_documents() -> list[tuple[str, str]]:
    docs_path = Path("docs/")
    if len(os.listdir(docs_path)) <= 2:
        pages = get_ducuments_from_url()
        for idx, page in enumerate(pages):
            with open(docs_path / f"{idx}.json", "w", encoding="utf-8") as f:
                json.dump(page, f, ensure_ascii=False)
    else:
        pages = get_documents_locally(dir=docs_path)
    title_abstract_docs = []
    for documents in pages:
        documents = documents["collection"]
        for doc in documents:
            title_abstract_docs.append((doc["title"], doc["abstract"]))
    return title_abstract_docs
