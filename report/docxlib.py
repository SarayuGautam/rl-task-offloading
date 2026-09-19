"""Minimal helpers for editing the report's word/document.xml in place with lxml.

The original report (a Google Docs export) uses direct run formatting on plain
paragraphs, so paragraphs are addressed by a unique substring of their text and
rewritten while keeping their paragraph properties and first-run formatting.
"""
import copy
import os
import re
import shutil
import subprocess
import zipfile

from lxml import etree

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def q(tag):
    p, local = tag.split(":")
    return f"{{{NS[p]}}}{local}"


class Docx:
    def __init__(self, src, workdir):
        if os.path.exists(workdir):
            shutil.rmtree(workdir)
        with zipfile.ZipFile(src) as z:
            z.extractall(workdir)
        self.dir = workdir
        self.doc_path = os.path.join(workdir, "word", "document.xml")
        self.tree = etree.parse(self.doc_path)
        self.body = self.tree.getroot().find(q("w:body"))
        self.rels_path = os.path.join(workdir, "word", "_rels", "document.xml.rels")
        self.rels = etree.parse(self.rels_path)

    # ------------------------------------------------------------------ queries
    def paras(self):
        return list(self.body.iter(q("w:p")))

    @staticmethod
    def text(p):
        return "".join(t.text or "" for t in p.iter(q("w:t")))

    def find(self, sub, nth=None, start=False, exact=False):
        def ok(t):
            if exact:
                return t.strip() == sub
            return t.startswith(sub) if start else sub in t
        hits = [p for p in self.paras() if ok(self.text(p))]
        if nth is None:
            assert len(hits) == 1, f"{len(hits)} paragraphs match {sub[:60]!r}"
            return hits[0]
        assert len(hits) > nth, f"only {len(hits)} paragraphs match {sub[:60]!r}"
        return hits[nth]

    @staticmethod
    def style(p):
        s = p.find("w:pPr/w:pStyle", NS)
        return s.get(q("w:val")) if s is not None else None

    # ------------------------------------------------------------------ editing
    @staticmethod
    def _first_rpr(p):
        for r in p.iter(q("w:r")):
            if r.find(q("w:t")) is not None:
                rpr = r.find(q("w:rPr"))
                return copy.deepcopy(rpr) if rpr is not None else None
        return None

    RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike",
                 "dstrike", "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid",
                 "vanish", "webHidden", "color", "spacing", "w", "kern", "position", "sz", "szCs",
                 "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs",
                 "em", "lang", "eastAsianLayout", "specVanish", "oMath"]

    @classmethod
    def make_run(cls, text, rpr=None, fmt=None):
        """Run with the base rPr plus overrides fmt = {b, i, font, sz, va}; children
        are re-sorted into schema order so Word accepts the file."""
        r = etree.Element(q("w:r"))
        rpr = copy.deepcopy(rpr) if rpr is not None else etree.Element(q("w:rPr"))
        fmt = fmt or {}

        def drop(*names):
            for n in names:
                for old in rpr.findall(q("w:" + n)):
                    rpr.remove(old)

        def put(name, **attrs):
            el = etree.SubElement(rpr, q("w:" + name))
            for k, v in attrs.items():
                el.set(q("w:" + k), v)

        for key, names in (("b", ("b", "bCs")), ("i", ("i", "iCs"))):
            if key in fmt:
                drop(*names)
                for n in names:
                    put(n, val="1" if fmt[key] else "0")
        if fmt.get("font"):
            drop("rFonts")
            put("rFonts", ascii=fmt["font"], hAnsi=fmt["font"], cs=fmt["font"], eastAsia=fmt["font"])
        if fmt.get("sz"):
            drop("sz", "szCs")
            put("sz", val=str(fmt["sz"]))
            put("szCs", val=str(fmt["sz"]))
        if fmt.get("u"):
            drop("u")
            put("u", val="single")
        if fmt.get("va"):
            drop("vertAlign")
            put("vertAlign", val=fmt["va"])
        order = {n: i for i, n in enumerate(cls.RPR_ORDER)}
        kids = sorted(list(rpr), key=lambda e: order.get(etree.QName(e).localname, 99))
        for k in list(rpr):
            rpr.remove(k)
        for k in kids:
            rpr.append(k)
        r.append(rpr)
        t = etree.SubElement(r, q("w:t"))
        t.text = text
        t.set(XML_SPACE, "preserve")
        return r

    def set_runs(self, p, segments):
        """Replace the paragraph content with runs [(text, fmt_dict), ...]."""
        base = self._first_rpr(p)
        keep = [c for c in p if c.tag in (q("w:pPr"), q("w:bookmarkStart"), q("w:bookmarkEnd"))]
        for c in list(p):
            p.remove(c)
        for c in keep:
            p.append(c)
        for text, fmt in segments:
            p.append(self.make_run(text, base, fmt))
        return p

    def set_text(self, p, text):
        return self.set_runs(p, [(text, {})])

    def replace_in(self, p, old, new, count=0):
        """Replace a substring inside run texts; falls back to rewriting the
        paragraph as one run when the substring spans runs."""
        done = False
        for t in p.iter(q("w:t")):
            if t.text and old in t.text:
                t.text = t.text.replace(old, new) if not count else t.text.replace(old, new, count)
                done = True
        if not done and old in self.text(p):
            self.set_text(p, self.text(p).replace(old, new))
            done = True
        return done

    def clone_after(self, anchor, template, segments=None, text=None):
        new = copy.deepcopy(template)
        for bm in list(new.iter(q("w:bookmarkStart"))) + list(new.iter(q("w:bookmarkEnd"))):
            bm.getparent().remove(bm)
        for d in list(new.iter(q("w:drawing"))):
            d.getparent().getparent().remove(d.getparent())
        if segments is not None:
            self.set_runs(new, segments)
        elif text is not None:
            self.set_text(new, text)
        anchor.addnext(new)
        return new

    @staticmethod
    def remove(el):
        el.getparent().remove(el)

    # ------------------------------------------------------------------ tables
    def tables(self):
        return list(self.body.iter(q("w:tbl")))

    @staticmethod
    def rows(tbl):
        return tbl.findall(q("w:tr"))

    def set_row(self, tr, values, fmts=None):
        cells = tr.findall(q("w:tc"))
        assert len(cells) == len(values), (len(cells), values)
        for i, (tc, v) in enumerate(zip(cells, values)):
            ps = tc.findall(q("w:p"))
            for extra in ps[1:]:
                tc.remove(extra)
            self.set_runs(ps[0], [(v, (fmts or {}).get(i, {}))])

    def add_row_after(self, tr, values, template=None, fmts=None):
        new = copy.deepcopy(template if template is not None else tr)
        self.set_row(new, values, fmts)
        tr.addnext(new)
        return new

    # ------------------------------------------------------------------ relationships
    def add_hyperlink_rel(self, url):
        root = self.rels.getroot()
        ids = {int(re.sub(r"\D", "", e.get("Id")) or 0) for e in root}
        rid = f"rId{max(ids) + 1}"
        el = etree.SubElement(root, f"{{{NS['rel']}}}Relationship")
        el.set("Id", rid)
        el.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink")
        el.set("Target", url)
        el.set("TargetMode", "External")
        return rid

    # ------------------------------------------------------------------ save
    def save(self, out):
        self.tree.write(self.doc_path, xml_declaration=True, encoding="UTF-8", standalone=True)
        self.rels.write(self.rels_path, xml_declaration=True, encoding="UTF-8", standalone=True)
        if os.path.exists(out):
            os.remove(out)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            # [Content_Types].xml first, as Word expects
            names = []
            for root, _, files in os.walk(self.dir):
                for f in files:
                    full = os.path.join(root, f)
                    names.append(os.path.relpath(full, self.dir))
            names.sort(key=lambda n: (n != "[Content_Types].xml", n))
            for n in names:
                z.write(os.path.join(self.dir, n), n)


# ---------------------------------------------------------------------- OMML math
def _m(tag, *children, **attrs):
    el = etree.Element(q("m:" + tag))
    for k, v in attrs.items():
        el.set(q("m:" + k), v)
    for c in children:
        el.append(c)
    return el


def mr(text, plain=False):
    r = _m("r")
    if plain:
        rpr = _m("rPr")
        rpr.append(_m("sty", val="p"))
        r.append(rpr)
    t = etree.SubElement(r, q("m:t"))
    t.text = text
    t.set(XML_SPACE, "preserve")
    return r


def msub(base, sub):
    return _m("sSub", _m("e", *base), _m("sub", *sub))


def msup(base, sup):
    return _m("sSup", _m("e", *base), _m("sup", *sup))


def mmax(lim, arg):
    """max over `lim` applied to `arg` (limit written below)."""
    return _m("func", _m("fName", _m("limLow", _m("e", mr("max", plain=True)), _m("lim", *lim))),
              _m("e", *arg))


def mdelim(inner, beg="[", end="]"):
    dpr = _m("dPr", _m("begChr", val=beg), _m("endChr", val=end))
    return _m("d", dpr, _m("e", *inner))


def math_paragraph(template_p, parts):
    """A centred display-equation paragraph holding one OMML expression."""
    p = etree.Element(q("w:p"))
    ppr = template_p.find(q("w:pPr"))
    if ppr is not None:
        p.append(copy.deepcopy(ppr))
    p.append(_m("oMathPara", _m("oMath", *parts)))
    return p


def render_pdf(docx_path, outdir):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", outdir, docx_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    return os.path.join(outdir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
