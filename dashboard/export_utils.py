# Bloomberg Dashboard Export Utilities

import os
import io
import base64
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from dash import html, dcc
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class BloombergExporter:
    """Bloomberg-style report generation and data export utilities."""
    
    def __init__(self, theme_config):
        self.theme = theme_config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def export_dashboard_pdf(self, portfolio_data: Dict[str, Any], 
                           risk_metrics: Dict[str, Any],
                           charts: Dict[str, go.Figure]) -> bytes:
        """Generate Bloomberg-style PDF report."""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        # Bloomberg-style formatting
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'BloombergTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.Color(248/255, 231/255, 28/255),  # Bloomberg yellow
            spaceAfter=20,
            fontName='Helvetica-Bold'
        )
        
        story = []
        
        # Header
        story.append(Paragraph("📈 BLOOMBERG TERMINAL ANALYTICS REPORT", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Portfolio Summary
        portfolio_table_data = [
            ['Metric', 'Value'],
            ['Total Assets', f"${portfolio_data.get('total_value', 0):,.2f}"],
            ['Return (1M)', f"{portfolio_data.get('monthly_return', 0):.2f}%"],
            ['Volatility', f"{portfolio_data.get('volatility', 0):.2f}%"],
            ['Sharpe Ratio', f"{portfolio_data.get('sharpe', 0):.3f}"]
        ]
        
        portfolio_table = Table(portfolio_table_data)
        portfolio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.1, 0.1, 0.1)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.Color(248/255, 231/255, 28/255)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.05, 0.05, 0.05)),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.3, 0.3, 0.3))
        ]))\n        \n        story.append(Paragraph("Portfolio Overview", styles['Heading2']))\n        story.append(portfolio_table)\n        story.append(Spacer(1, 20))\n        \n        # Risk Metrics\n        risk_table_data = [\n            ['Risk Metric', 'Value'],\n            ['VaR (95%)', f\"{risk_metrics.get('var_95', 0):.2f}%\"],\n            ['Expected Shortfall', f\"{risk_metrics.get('es_95', 0):.2f}%\"],\n            ['Maximum Drawdown', f\"{risk_metrics.get('max_drawdown', 0):.2f}%\"],\n            ['Beta', f\"{risk_metrics.get('beta', 0):.3f}\"]\n        ]\n        \n        risk_table = Table(risk_table_data)\n        risk_table.setStyle(TableStyle([\n            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.1, 0.1, 0.1)),\n            ('TEXTCOLOR', (0, 0), (-1, 0), colors.Color(0, 230/255, 255/255)),  # Bloomberg cyan\n            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),\n            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),\n            ('FONTSIZE', (0, 0), (-1, 0), 12),\n            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),\n            ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.05, 0.05, 0.05)),\n            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),\n            ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.3, 0.3, 0.3))\n        ]))\n        \n        story.append(Paragraph("Risk Analytics", styles['Heading2']))\n        story.append(risk_table)\n        \n        # Build PDF\n        doc.build(story)\n        buffer.seek(0)\n        return buffer.getvalue()\n    \n    def export_excel_report(self, data_dict: Dict[str, pd.DataFrame]) -> bytes:\n        \"\"\"Export comprehensive Excel workbook with all data.\"\"\"\n        \n        buffer = io.BytesIO()\n        \n        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:\n            # Write each dataset to separate sheets\n            for sheet_name, df in data_dict.items():\n                df.to_excel(writer, sheet_name=sheet_name, index=False)\n                \n                # Format Bloomberg-style\n                worksheet = writer.sheets[sheet_name]\n                \n                # Header formatting\n                for col in range(1, len(df.columns) + 1):\n                    cell = worksheet.cell(row=1, column=col)\n                    cell.fill = openpyxl.styles.PatternFill(\n                        start_color=\"F8E71C\", end_color=\"F8E71C\", fill_type=\"solid\"\n                    )\n                    cell.font = openpyxl.styles.Font(bold=True, color=\"000000\")\n        \n        buffer.seek(0)\n        return buffer.getvalue()\n    \n    def create_download_button(self, file_data: bytes, filename: str, button_text: str) -> html.Div:\n        \"\"\"Create Bloomberg-style download button.\"\"\"\n        \n        encoded_data = base64.b64encode(file_data).decode()\n        \n        return html.Div([\n            html.A(\n                button_text,\n                href=f\"data:application/octet-stream;base64,{encoded_data}\",\n                download=filename,\n                style={\n                    \"backgroundColor\": self.theme[\"accent\"],\n                    \"color\": \"#000000\",\n                    \"padding\": \"12px 24px\",\n                    \"textDecoration\": \"none\",\n                    \"borderRadius\": \"4px\",\n                    \"fontWeight\": \"bold\",\n                    \"fontSize\": \"12px\",\n                    \"textTransform\": \"uppercase\",\n                    \"letterSpacing\": \"0.5px\",\n                    \"display\": \"inline-block\",\n                    \"margin\": \"8px\",\n                    \"border\": \"none\",\n                    \"cursor\": \"pointer\",\n                    \"transition\": \"all 0.2s ease\"\n                }\n            )\n        ])\n    \n    def export_chart_image(self, figure: go.Figure, width: int = 1200, height: int = 800) -> bytes:\n        \"\"\"Export Plotly chart as high-resolution image.\"\"\"\n        \n        # Configure for Bloomberg-style export\n        figure.update_layout(\n            width=width,\n            height=height,\n            font=dict(family=\"'Bloomberg', monospace\", size=12),\n            paper_bgcolor=self.theme[\"bg\"],\n            plot_bgcolor=self.theme[\"panel\"]\n        )\n        \n        return figure.to_image(format=\"png\", scale=2)\n\ndef create_export_panel(theme_config) -> html.Div:\n    \"\"\"Create Bloomberg-style export control panel.\"\"\"\n    \n    return html.Div([\n        html.H4(\"📊 EXPORT CONTROLS\", style={\n            \"color\": theme_config[\"accent\"],\n            \"fontSize\": \"14px\",\n            \"fontWeight\": \"700\",\n            \"textTransform\": \"uppercase\",\n            \"letterSpacing\": \"0.8px\",\n            \"marginBottom\": \"16px\"\n        }),\n        \n        html.Div([\n            html.Button(\"📄 Generate PDF Report\", \n                       id=\"export-pdf-btn\",\n                       className=\"bloomberg-export-btn\"),\n            html.Button(\"📊 Export Excel Data\", \n                       id=\"export-excel-btn\",\n                       className=\"bloomberg-export-btn\"),\n            html.Button(\"📈 Save All Charts\", \n                       id=\"export-charts-btn\",\n                       className=\"bloomberg-export-btn\"),\n        ], style={\"display\": \"flex\", \"gap\": \"12px\", \"flexWrap\": \"wrap\"}),\n        \n        html.Div(id=\"export-status\", style={\n            \"marginTop\": \"12px\",\n            \"color\": theme_config[\"muted\"],\n            \"fontSize\": \"11px\"\n        }),\n        \n        html.Div(id=\"export-downloads\")\n        \n    ], style={\n        \"padding\": \"20px\",\n        \"backgroundColor\": theme_config[\"card\"],\n        \"border\": f\"1px solid {theme_config['border']}\",\n        \"borderRadius\": \"6px\",\n        \"marginTop\": \"16px\"\n    })