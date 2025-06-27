import os
import re
import json
from PyPDF2 import PdfReader

def extract_previas_from_pdf(pdf_path: str, concurso_id: int) -> list:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() for page in reader.pages]
    full_text = "\n".join(pages)

    # Patrón por partido (divide texto por bloques de partidos)
    partidos_raw = re.split(r'\bPartido\s+\d+\b', full_text)[1:]  # omite encabezado

    previas = []
    for idx, bloque in enumerate(partidos_raw, start=1):
        previa = {
            "match_id": f"{concurso_id}-{idx}",
            "form_H": None,
            "form_A": None,
            "h2h_H": 0,
            "h2h_E": 0,
            "h2h_A": 0,
            "inj_H": 0,
            "inj_A": 0,
            "context_flag": []
        }

        # Forma
        form_match = re.findall(r'([WDL])\s\d-\d', bloque)
        if len(form_match) >= 10:
            previa["form_H"] = "".join(form_match[:5])
            previa["form_A"] = "".join(form_match[5:10])

        # H2H (e.g. "H2H: 3V 1E 1D")
        h2h = re.findall(r'H2H.*?(\d+)[^\d]*(\d+)[^\d]*(\d+)', bloque)
        if h2h:
            h, e, v = map(int, h2h[0])
            previa["h2h_H"], previa["h2h_E"], previa["h2h_A"] = h, e, v

        # Lesiones y dudas (simplificado, puede mejorar)
        previa["inj_H"] = len(re.findall(r'(fuera|lesión|duda).+?(local|casa)', bloque, re.IGNORECASE))
        previa["inj_A"] = len(re.findall(r'(fuera|lesión|duda).+?(visita|visitante)', bloque, re.IGNORECASE))

        # Flags contextuales
        if re.search(r'\bfinal\b', bloque, re.IGNORECASE):
            previa["context_flag"].append("final")
        if re.search(r'\bderbi\b|\bclásico\b', bloque, re.IGNORECASE):
            previa["context_flag"].append("derbi")

        previas.append(previa)

    return previas

def guardar_previas_json(previas: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(previas, f, ensure_ascii=False, indent=2)

# Ejemplo de uso
if __name__ == "__main__":
    previas = extract_previas_from_pdf("data/raw/previas_2283.pdf", concurso_id=2283)
    guardar_previas_json(previas, "data/json_previas/previas_2283.json")
