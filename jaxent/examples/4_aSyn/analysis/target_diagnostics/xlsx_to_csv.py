"""Dump selected worksheets of an xlsx to CSV (stdlib only)."""
import zipfile, csv, sys, xml.etree.ElementTree as ET
from pathlib import Path
src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(exist_ok=True)
z = zipfile.ZipFile(src); names = z.namelist()
ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
wb = ET.fromstring(z.read("xl/workbook.xml"))
rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
rid2file = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
ss = []
if "xl/sharedStrings.xml" in names:
    for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
        ss.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
def col_index(ref):
    s = "".join(ch for ch in ref if ch.isalpha()); n = 0
    for ch in s: n = n * 26 + ord(ch) - 64
    return n - 1
for s in wb.find("m:sheets", ns):
    name = s.attrib["name"]; f = rid2file[s.attrib["{%s}id" % ns["r"]]]
    f = f if f.startswith("xl/") else "xl/" + f
    rows = ET.fromstring(z.read(f)).find("m:sheetData", ns)
    table = []
    for r in rows:
        vals = {}
        for c in r:
            v = c.find("m:v", ns)
            if v is None: continue
            vals[col_index(c.attrib["r"])] = ss[int(v.text)] if c.attrib.get("t") == "s" else v.text
        width = max(vals) + 1 if vals else 0
        table.append([vals.get(i, "") for i in range(width)])
    safe = name.replace(" ", "_").replace("/", "_")
    with open(out / f"{safe}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(table)
    print(f"{name:24s} -> {safe}.csv rows {len(table)} maxcols {max((len(t) for t in table), default=0)}")
