import os

from aleph_alpha_client import Client, Document, SummarizationRequest
from dotenv import load_dotenv

load_dotenv()

client = Client(token=os.getenv("AA_TOKEN"))

with open("ressources/einstein.txt") as f:
    prompt_text = f.read()


doc = Document.from_text(prompt_text)

request = SummarizationRequest(doc)
response = client.summarize(request=request)
summary = response.summary
print(summary)
