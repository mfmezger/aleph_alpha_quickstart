from aleph_alpha_client import Client, Prompt, SummarizationRequest, Document
import os
from dotenv import load_dotenv

load_dotenv()

client = Client(token=os.getenv("AA_TOKEN"))


prompt_text = """Albert Einstein (* 14. März 1879 in Ulm, Königreich Württemberg; † 18. April 1955 in Princeton, New Jersey) war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft. Der Wissenschaftler jüdischer Abstammung hatte ab 1901 die Schweizer und ab 1940 zusätzlich die US-amerikanische Staatsbürgerschaft. Deutscher Staatsangehöriger war Einstein nochmals von 1914 bis 1934.

Er gilt als einer der bedeutendsten Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit. Seine Forschungen zur Struktur von Materie, Raum und Zeit sowie zum Wesen der Gravitation veränderten maßgeblich das zuvor geltende newtonsche Weltbild.

Einsteins Hauptwerk, die Relativitätstheorie, machte ihn weltberühmt. Im Jahr 1905 erschien seine Arbeit mit dem Titel Zur Elektrodynamik bewegter Körper, deren Inhalt heute als Spezielle Relativitätstheorie bezeichnet wird. 1915 publizierte er die Allgemeine Relativitätstheorie. Auch zur Quantenphysik leistete er wesentliche Beiträge. „Für seine Verdienste um die Theoretische Physik, besonders für seine Entdeckung des Gesetzes des photoelektrischen Effekts“, erhielt er den Nobelpreis des Jahres 1921, der ihm 1922 überreicht wurde. Seine theoretischen Arbeiten spielten – im Gegensatz zur weit verbreiteten Meinung – beim Bau der Atombombe und der Entwicklung der Kernenergie nur eine indirekte Rolle.[1]

Albert Einstein gilt als Inbegriff des Forschers und Genies. Er nutzte seine außerordentliche Bekanntheit auch außerhalb der naturwissenschaftlichen Fachwelt bei seinem Einsatz für Völkerverständigung, Frieden und Sozialismus.[2]"""

doc = Document.from_text(prompt_text)

request = SummarizationRequest(doc)
response = client.summarize(request=request)
summary = response.summary
print(summary)
