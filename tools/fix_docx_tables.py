"""Quarto appends a second <w:jc center/> after each table cell's own alignment; Word and
Google Docs apply the last one, centering every cell. Keep the first alignment only, and
left-align the single-cell wrapper table that holds the caption."""
import re, shutil, sys, zipfile
src = sys.argv[1]; tmp = src + ".tmp"
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            s = data.decode("utf8")
            n = len(re.findall(r'(<w:jc w:val="(?:left|right|center)" ?/>)\s*<w:jc w:val="center" ?/>', s))
            s = re.sub(r'(<w:jc w:val="(?:left|right|center)" ?/>)\s*<w:jc w:val="center" ?/>', r"\1", s)
            s = s.replace('<w:pPr>\n<w:jc w:val="center" />\n</w:pPr>\n<w:pPr>', "<w:pPr>")
            print("fixed cells:", n)
            data = s.encode("utf8")
        zout.writestr(item, data)
shutil.move(tmp, src)
