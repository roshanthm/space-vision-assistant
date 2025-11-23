from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf_report(label, conf, explanation):
    file_path = "report.pdf"
    c = canvas.Canvas(file_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, 750, "Space Vision Report")

    c.setFont("Helvetica", 14)
    c.drawString(30, 700, f"Detected Object: {label}")
    c.drawString(30, 680, f"Confidence: {conf*100:.2f}%")

    c.setFont("Helvetica", 12)
    c.drawString(30, 640, "AI Explanation:")
    text = c.beginText(30, 620)
    text.setFont("Helvetica", 11)
    for line in explanation.split("\n"):
        text.textLine(line)

    c.drawText(text)
    c.save()

    return file_path
