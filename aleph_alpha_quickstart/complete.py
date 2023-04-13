import os

from aleph_alpha_client import Client, CompletionRequest, Prompt
from dotenv import load_dotenv

load_dotenv()

model = Client(token=os.getenv("AA_TOKEN"))


prompt_text = """
Please extract information from the text into a markdown table.
###
Text:
RECHNUNGMax Musterman — Musterstr. X — 12345 Musterstadt Firmenname Empfänger Rechnungsdatum: 01.01.2020
Name EmpfängerStraßenname 345PLZ OrtLand (optional)01.01.2020
RECHNUNGSehr geehrte Damen und Herren,im folgenden der/die erworbene/n Artikel, die ich in Rechnung stelle:Pos. Bezeichnung Menge Einzelpreis Gesamtpreis
1. Gebrauchtes Mountainbike 1,00 Stk 140,00 € 140,00 €
2. Gebrauchter Fahrradhelm 1,00 Stk 20,00 € 20,00 €
Summe Positionen 160,00 €
Rechnungsbetrag 160,00 €Zahlungsbedingungen: Barzahlung bei Abholung.      Mit freundlichen GrüßenNılak,Max MusterDiese Vorlage
wurde erstellt von:
###
Please extract the information Name, Adresse, Rechnungsdatum, Artikel 0 ... ArtikelX, Gesamtpreis
Table:
|Key|Value|
|Name| Max Musterman|
|Adresse|Musterstraße X - 12345 Musterstadt|
|Rechnungsdatum|01.01.2020|
|Artikel0|Gebrauchtes Mountain bike|
|Artikel1|Gebrauchter Fahrradhelm|
|Gesamtpreis|160€|
###
Text:
dd Muster GmbHMuster GmbH Lange Str. 2 | 10245 BerlinHabermann & SöhneSchnurlos-Straße 81
34131 KasselMuster GmbH
Lange Str. 2
10245 Berlin+49 (0) 30 2121356mail@muster.de
www.muster.de  Rechnung
Rechnungs-Nr.: M1675 Rechnungsdatum: 29.10.20
Auftrags-Nr.: 01727 Lieferdatum: 30.10.20
Komission: Bestellung Gerstner Bearbeiter: Dorothea Schäfer
Kunden-Nr. 1068 Telefon: 030 2121359
Bestell-Nr.: 369852 E-Mail: mail@muster.de
Pos. Art-Nr. Bezeichnung Menge Einheit Preis/Einh. (€) Gesamt (€)
1 B-3025-078 B-3025, Farbe Grün 1,00 Stk. 47,00 47,00
Musterartikel
2 B-0050-050 B-0050, Farbe Blau 2,00 Stk. 36,00 72,00
Musterartikel ABC
3 A-0086-007 A-0086, Antik-Look 1,00 Stk. 56,00 56,00
Musterartikel
3 V-13kg Versand und Verpackung 1,00 Stk. 11,99 11,99
Summe Netto € 186,99
19,00% USt. auf 186,99 € € 35,53
Endsumme € 222,51Lieferbedingung: Postversand10 Tage 5% Skonto, 30 Tage ohne AbzugMuster GmbH : Sparkasse Berlin :
Konto 10 25 25 25 : BLZ 500 600 26 : IBAN DE10 25 25 25 500 600 26 02: BIC HERAKLES02
Sitz der Gesellschaft: Berlin, Deutschland : Geschäftsführung: Max Mustermann - Handelsregister: AG Berlin HRB 123456 - USt-IdNr. DE216398573
Diese Rechnungsvorlage wurde erstellt von www.weclapp.com/de
###
Please extract the information Name, Adresse, Rechnungsdatum, Artikel 0 ... ArtikelX, Gesamtpreis
Table:
|Key|Value|
"""

params = {
    "prompt": Prompt.from_text(prompt_text),
    "maximum_tokens": 230,
    "temperature": 0,
    "stop_sequences": ["###"],
}

request = CompletionRequest(**params)
response = model.complete(request, model="luminous-extended")

print(f"\nKeywords: {response.completions[0].completion}")
