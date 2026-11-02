"""Validation for Qwen few-shot robot annotations."""
from __future__ import annotations
import json, re

PROMPT = """The preceding images are labeled examples. Red outlines mean red
robots; blue outlines mean blue robots. Do not label towers, field structures,
people, score graphics, balls, reflections, or shadows. For the final TARGET
image, return only clearly visible robots as exact bounding boxes.
Return exactly {"boxes":[{"alliance":"red|blue","box":[x1,y1,x2,y2],
"confidence":0.0,"evidence":"brief"}]}. Coordinates are 0..1000. Use an
empty boxes list if uncertain; never guess."""

def parse(text):
    try: return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I|re.S)), None
    except Exception as error: return None, str(error)

def normalize(value):
    if not isinstance(value, dict) or not isinstance(value.get("boxes"), list): return None, "response lacks boxes"
    output=[]
    for item in value["boxes"]:
        if not isinstance(item,dict) or item.get("alliance") not in {"red","blue"}: continue
        try: x1,y1,x2,y2=[max(0,min(1000,round(float(n)))) for n in item["box"]]
        except Exception: continue
        if x2-x1 < 15 or y2-y1 < 15: continue
        output.append({"alliance":item["alliance"],"box":[x1,y1,x2,y2],"confidence":max(0,min(1,float(item.get("confidence",0)))),"evidence":str(item.get("evidence", ""))[:300]})
    return output,None
