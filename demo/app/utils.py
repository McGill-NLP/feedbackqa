import json
import os
from typing import List, Dict

from bs4 import BeautifulSoup
import tomd
import random
from random import sample
from collections import defaultdict


def preprocess_html(html: str, domain: str) -> str:
    """
    Fix some <a> links
    """
    html = html.replace("\n", " ").replace("external icon", "").replace("pdf icon", "")
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a"):
        href = a.get("href")
        if href.startswith("/"):
            href = domain + href
        href = href.replace(" ", "")
        a["href"] = href

    return str(soup)


def convert_to_md(html: str, domain: str = "https://www.cdc.gov") -> str:
    """
    Pre-process and convert from html to markdown
    """
    html = preprocess_html(html, domain=domain)
    md = tomd.convert(html).replace(" - ", "\n- ")
    return md

def sample_questions(
    load_dir: str, 
    regions: List[str] = ["Australia", "UK", "CDC", "WHO"],
    n_samples: int = 20, 
    split: str = "train") -> Dict:
    
    questions = {}
    for region in regions:
        load_path = os.path.join(load_dir, region, "feedback_" + split + ".json")
        data = json.load(open(load_path, "r"))
        ques = [x['question'] for x in data]
        ques = list(set(ques))
        random.shuffle(ques)
        questions[region] = ques#sample(ques, n_samples)
    return questions


def preprocess_crowdsource_data(
    load_dir: str,
    save_dir: str,
    regions: List[str] = ["Australia", "UK", "CDC", "WHO"],
    splits: List[str] = ["train"],
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for region in regions:
        passages_raw = []
        for dsplit in splits:
            load_path = os.path.join(load_dir, region, dsplit + ".json")
            data = json.load(open(load_path, "r"))
            passages_raw.extend(data["passages"])

        passages = [
            {
                "source": p["source"],
                "uri": p["uri"],
                "content": p["reference"]["section_content"],
                "content_html": p["reference"]["section_content_html"],
            }
            for p in passages_raw
        ]

        candidates = [p["content"] for p in passages]

        save_path = os.path.join(save_dir, f"passages_{region}.json")
        json.dump(passages, open(save_path, "w"))
