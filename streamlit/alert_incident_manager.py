import os
import time
from datetime import datetime
from PIL import Image
from fpdf import FPDF
import cv2


def create_pdf_report(image_files, output_pdf_path, images_per_page=6):
    """Create a PDF report with 6 images per page from the captured alert frames"""
    if not image_files:
        return None
    
    # Create PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add title page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 20, "Drowning Detection Alert Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 10, f"Total Alert Frames: {len(image_files)}", ln=True, align="C")
    
    pdf.add_page()
    
    # Calculate dimensions
    page_width = pdf.w - 2 * pdf.l_margin
    page_height = pdf.h - 2 * pdf.t_margin
    
    # For 2x3 grid layout
    cols = 2
    rows = 3
    
    # Calculate dimensions for each image cell
    cell_width = page_width / cols
    cell_height = page_height / rows
    
    # Add images
    image_count = 0
    for i, img_path in enumerate(image_files):
        # Add new page when needed
        if i % images_per_page == 0 and i > 0:
            pdf.add_page()
        
        # Calculate position
        col = (i % images_per_page) % cols
        row = (i % images_per_page) // cols
        
        x = pdf.l_margin + col * cell_width
        y = pdf.t_margin + row * cell_height
        
        # Add image
        try:
            # Open with PIL to get dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                # Calculate scaling to fit in cell
                img_ratio = width / height
                cell_ratio = cell_width / cell_height
                
                if img_ratio > cell_ratio:  # Image is wider than cell
                    img_width = cell_width - 10  # Margin
                    img_height = img_width / img_ratio
                else:  # Image is taller than cell
                    img_height = cell_height - 20  # Margin plus space for caption
                    img_width = img_height * img_ratio
                
                # Center the image in the cell
                x_centered = x + (cell_width - img_width) / 2
                
                pdf.image(img_path, x=x_centered, y=y, w=img_width, h=img_height)
                
                # Add timestamp caption
                timestamp = os.path.basename(img_path).split('_')[-1].split('.')[0]
                pdf.set_font("Arial", "", 8)
                pdf.text(x + 5, y + img_height + 10, f"Frame #{timestamp}")
                
                image_count += 1
        except Exception as e:
            print(f"Error adding image {img_path} to PDF: {e}")
    
    # Save PDF
    pdf.output(output_pdf_path)
    return output_pdf_path

def save_alert_frames(frame, alert_folder, session_id, frame_count):
    """Save frame to the alert folder with appropriate naming"""
    frame_filename = os.path.join(alert_folder, f"alert_{session_id}_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    return frame_filename


class AlertIncidentManager:
    def __init__(self, base_dir, session_id):
        self.base_dir = base_dir
        self.session_id = session_id
        self.incidents = {}  # track_id -> dict with capture info

    def start_incident(self, track_id):
        if track_id not in self.incidents:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            incident_dir = os.path.join(self.base_dir, f"alertIMG_{self.session_id}_{track_id}_{timestamp}")
            os.makedirs(incident_dir, exist_ok=True)
            self.incidents[track_id] = {
                "start_time": time.time(),
                "frame_count": 0,
                "frames": [],
                "dir": incident_dir,
                "pdf_created": False
            }
        
    def generate_pdf(self, track_id):
        incident = self.incidents[track_id]
        if not incident["pdf_created"]:
            incident["pdf_created"] = True
            pdf_path = self._generate_pdf_sync(track_id, incident["frames"])  # You can make this synchronous
            incident["pdf_path"] = pdf_path
            return pdf_path
        return incident.get("pdf_path", None)
    
    def _generate_pdf_sync(self, track_id, frame_paths):
        output_pdf_dir = os.path.join(self.base_dir, f"alertPDF_{self.session_id}")
        if not os.path.exists(output_pdf_dir):
            os.makedirs(output_pdf_dir)

        output_pdf_path = os.path.join(output_pdf_dir, f"alert_report_{track_id}.pdf")
        return create_pdf_report(frame_paths, output_pdf_path)


    def capture_frame_only(self, track_id, frame):
        incident = self.incidents[track_id]
        elapsed = time.time() - incident["start_time"]

        if elapsed <= 5:
            frame_path = save_alert_frames(
                frame, 
                incident["dir"],
                self.session_id,
                incident["frame_count"]
            )
            incident["frames"].append(frame_path)
            incident["frame_count"] += 1

