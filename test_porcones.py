from pdf2image import convert_from_path

poppler = "D:/AI/Project/Release-25.12.0-0/poppler-25.12.0/Library/bin"

files = [
    "data/Print/PORCONES.228.38 \u2013 1646.pdf",
    "data/Print/PORCONES.23.5 - 1628.pdf",
    "data/Print/PORCONES.748.6 \u2013 1650.pdf",
]

for f in files:
    try:
        imgs = convert_from_path(f, dpi=72, first_page=1, last_page=1, poppler_path=poppler)
        print(f"OK: {f} -> size {imgs[0].size}")
    except Exception as e:
        print(f"ERROR: {f} -> {e}")