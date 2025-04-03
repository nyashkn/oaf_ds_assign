"""
Model Comparison Report Generator

This module generates a comprehensive report showing the comparison
between two loan repayment prediction models:
1. Model 1: Using only application-time features
2. Model 2: Including September payment data

The report includes:
- Performance metrics comparison
- Profitability analysis at different thresholds
- ROI analysis
- Feature importance comparison
- Business recommendations
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, List, Optional, Tuple, Any, Union

# Set plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

class ModelComparisonReport:
    """Class to generate a comprehensive model comparison report."""
    
    def __init__(
        self, 
        data_dir: str = "data/processed/model_comparison",
        output_dir: str = "reports"
    ):
        """
        Initialize the report generator.
        
        Args:
            data_dir: Directory containing model comparison results
            output_dir: Directory to save the report
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load comparison data
        self.comparison_data = self._load_comparison_data()
        self.model1_metrics = self._load_metrics("model1_threshold_metrics.csv")
        self.model2_metrics = self._load_metrics("model2_threshold_metrics.csv")
        self.model1_importance = self._load_feature_importance("model1_feature_importance.csv")
        self.model2_importance = self._load_feature_importance("model2_feature_importance.csv")
        self.model1_tradeoffs = self._load_tradeoffs("model1_tradeoffs.json")
        self.model2_tradeoffs = self._load_tradeoffs("model2_tradeoffs.json")
        
        # Set up PDF document
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        
        # Use standard Helvetica font
        self.pdf.set_font('Helvetica', '', 11)
        
    def _load_comparison_data(self) -> Dict:
        """Load model comparison data."""
        try:
            with open(os.path.join(self.data_dir, "model_comparison.json"), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: model_comparison.json not found in {self.data_dir}")
            return {}
            
    def _load_metrics(self, filename: str) -> pd.DataFrame:
        """Load threshold metrics data."""
        try:
            return pd.read_csv(os.path.join(self.data_dir, filename))
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {self.data_dir}")
            return pd.DataFrame()
            
    def _load_feature_importance(self, filename: str) -> pd.DataFrame:
        """Load feature importance data."""
        try:
            return pd.read_csv(os.path.join(self.data_dir, filename))
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {self.data_dir}")
            return pd.DataFrame()
            
    def _load_tradeoffs(self, filename: str) -> Dict:
        """Load tradeoff analysis data."""
        try:
            with open(os.path.join(self.data_dir, filename), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {self.data_dir}")
            return {}
    
    def add_title(self, title: str, subtitle: Optional[str] = None):
        """Add a title to the report."""
        self.pdf.set_font('Helvetica', 'B', 22)
        self.pdf.cell(0, 10, title, ln=True, align='C')
        
        if subtitle:
            self.pdf.set_font('Helvetica', '', 14)
            self.pdf.ln(5)
            self.pdf.cell(0, 10, subtitle, ln=True, align='C')
        
        self.pdf.ln(5)
        
    def add_section_header(self, title: str):
        """Add a section header to the report."""
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.ln(5)
        self.pdf.cell(0, 10, title, ln=True)
        self.pdf.ln(2)
        
    def add_subsection_header(self, title: str):
        """Add a subsection header to the report."""
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.ln(3)
        self.pdf.cell(0, 8, title, ln=True)
        self.pdf.ln(1)
        
    def add_paragraph(self, text: str):
        """Add a paragraph to the report."""
        self.pdf.set_font('Helvetica', '', 11)
        self.pdf.multi_cell(0, 6, text)
        self.pdf.ln(2)
        
    def add_table(self, headers: List[str], data: List[List[str]], column_widths: Optional[List[int]] = None):
        """Add a table to the report."""
        self.pdf.set_font('Helvetica', 'B', 10)
        
        # Set default column widths if not provided
        if column_widths is None:
            column_widths = [180 // len(headers)] * len(headers)
        
        # Table headers
        for i, header in enumerate(headers):
            self.pdf.cell(column_widths[i], 7, header, 1, 0, 'C')
        self.pdf.ln()
        
        # Table data
        self.pdf.set_font('Helvetica', '', 10)
        for row in data:
            for i, cell in enumerate(row):
                self.pdf.cell(column_widths[i], 6, cell, 1, 0, 'C')
            self.pdf.ln()
        
        self.pdf.ln(3)
    
    def add_image(self, image_path: str, w: int = 190, h: Optional[int] = None, caption: Optional[str] = None):
        """Add an image to the report."""
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            return
            
        try:
            # Add the image
            if h:
                self.pdf.image(image_path, x=10, y=self.pdf.get_y(), w=w, h=h)
                self.pdf.ln(h + 5)  # Add space after the image
            else:
                self.pdf.image(image_path, x=10, y=self.pdf.get_y(), w=w)
                # Estimate height based on aspect ratio
                img_height = w * 0.6  # Approximate height based on width
                self.pdf.ln(img_height + 5)  # Add space after the image
                
            # Add caption if provided
            if caption:
                self.pdf.set_font('Helvetica', '', 9)
                self.pdf.cell(0, 5, caption, ln=True, align='C')
                self.pdf.ln(2)
        except Exception as e:
            print(f"Error adding image {image_path}: {e}")
    
    def create_executive_summary(self):
        """Create the executive summary section."""
        self.add_section_header("1. Executive Summary")
        
        if not self.comparison_data:
            self.add_paragraph("Error: Comparison data not available. Please run the model comparison first.")
            return
            
        # Extract metrics
        perf_imp = self.comparison_data.get('performance_improvement', {})
        rmse_imp = perf_imp.get('rmse', 0)
        r2_imp = perf_imp.get('r2', 0)
        profit_imp = self.comparison_data.get('profit_improvement', 0)
        roi_imp = self.comparison_data.get('roi_improvement', 0)
        
        model1_optimal = self.comparison_data.get('model1_optimal', {})
        model2_optimal = self.comparison_data.get('model2_optimal', {})
        
        # Create summary text
        summary = (
            "This report compares two approaches for loan repayment rate prediction. "
            "Model 1 uses only application-time features, while Model 2 incorporates "
            "September payment data in addition to application-time features.\n\n"
            
            f"Model 2 shows a {abs(rmse_imp):.1f}% {'decrease' if rmse_imp < 0 else 'increase'} in RMSE and a "
            f"{abs(r2_imp):.1f}% {'increase' if r2_imp > 0 else 'decrease'} in R² compared to Model 1. "
            f"This demonstrates the {'positive' if rmse_imp < 0 and r2_imp > 0 else 'mixed'} impact of "
            "including September payment data on predictive accuracy.\n\n"
            
            f"In terms of profitability, Model 2 shows a {abs(profit_imp):.1f}% {'increase' if profit_imp > 0 else 'decrease'} "
            f"in profit and a {abs(roi_imp):.1f}% {'increase' if roi_imp > 0 else 'decrease'} in ROI at their respective "
            "optimal thresholds. This highlights the business value of incorporating payment data when available."
        )
        
        self.add_paragraph(summary)
        
        # Add key metrics table
        self.add_subsection_header("Key Performance Indicators")
        
        # Get optimal thresholds
        model1_profit_threshold = model1_optimal.get('profit', {}).get('threshold', 0)
        model2_profit_threshold = model2_optimal.get('profit', {}).get('threshold', 0)
        
        model1_profit_value = model1_optimal.get('profit', {}).get('value', 0)
        model2_profit_value = model2_optimal.get('profit', {}).get('value', 0)
        
        model1_roi_value = model1_optimal.get('roi', {}).get('value', 0)
        model2_roi_value = model2_optimal.get('roi', {}).get('value', 0)
        
        headers = ["Metric", "Model 1", "Model 2", "Improvement"]
        data = [
            ["Accuracy", f"{self.model1_metrics['accuracy'].mean():.4f}", f"{self.model2_metrics['accuracy'].mean():.4f}", f"{rmse_imp:.1f}%"],
            ["ROI", f"{self.model1_metrics['roi'].mean():.4f}", f"{self.model2_metrics['roi'].mean():.4f}", f"{r2_imp:.1f}%"],
            ["Optimal Threshold (Profit)", f"{model1_profit_threshold:.2f}", f"{model2_profit_threshold:.2f}", "N/A"],
            ["Max Profit", f"{model1_profit_value:.0f}", f"{model2_profit_value:.0f}", f"{profit_imp:.1f}%"],
            ["Max ROI", f"{model1_roi_value:.4f}", f"{model2_roi_value:.4f}", f"{roi_imp:.1f}%"]
        ]
        
        self.add_table(headers, data)
        
        # Business Recommendations
        self.add_subsection_header("Business Recommendations")
        
        if profit_imp > 0 and roi_imp > 0:
            recommendation = (
                "Based on the analysis, we recommend using Model 2 (with September payment data) "
                "for loan approval decisions when this information is available. This would maximize "
                "both profits and ROI.\n\n"
                
                "For new loan applications where September payment data is not yet available, "
                "Model 1 should be used with the recommended threshold of "
                f"{model1_profit_threshold:.2f} for profit-focused decisions or a higher threshold "
                "if ROI is the primary concern."
            )
        else:
            recommendation = (
                "The analysis shows mixed results between the two models. While Model 2 offers "
                f"{'better' if rmse_imp < 0 else 'worse'} predictive accuracy, it results in "
                f"{'higher' if profit_imp > 0 else 'lower'} profits and "
                f"{'higher' if roi_imp > 0 else 'lower'} ROI.\n\n"
                
                "We recommend conducting further analysis to understand these tradeoffs better "
                "before making a final decision on which model to implement for loan approvals."
            )
            
        self.add_paragraph(recommendation)
    
    def create_model_descriptions(self):
        """Create the model descriptions section."""
        self.add_section_header("2. Model Descriptions")
        
        # Model 1 description
        self.add_subsection_header("Model 1: Application-Time Features Only")
        
        model1_desc = (
            "This model uses only features available at the time of loan application, "
            "making it suitable for new loan application decisions when no payment history exists. "
            "The features include:\n\n"
            "- Basic loan details (loan type, region, duka name, etc.)\n"
            "- Contract value and deposit amount\n"
            "- Historical metrics by region and territory\n"
            "- Temporal features (contract start date, day of week, etc.)\n"
            "- Geographic features (distances to key locations)\n\n"
            
            "This model represents a production-ready approach that can be used for all new loan applications."
        )
        
        self.add_paragraph(model1_desc)
        
        # Add top features for Model 1
        if not self.model1_importance.empty:
            top_features_1 = self.model1_importance.head(5)
            
            headers = ["Feature", "Importance"]
            data = [[row['feature'], f"{row['importance']:.4f}"] for _, row in top_features_1.iterrows()]
            
            self.add_subsection_header("Top Features - Model 1")
            self.add_table(headers, data)
        
        # Model 2 description
        self.add_subsection_header("Model 2: Including September Payment Data")
        
        model2_desc = (
            "This model includes all application-time features plus data from the September payment period. "
            "The additional features include:\n\n"
            "- Cumulative amount paid up to September\n"
            "- September repayment rate\n"
            "- Payment velocity (payment per day)\n"
            "- Days between contract start and September\n\n"
            
            "This model can be used for ongoing risk assessment and portfolio management for existing loans "
            "where payment history is available. It provides better predictive performance but requires "
            "post-application data."
        )
        
        self.add_paragraph(model2_desc)
        
        # Add top features for Model 2
        if not self.model2_importance.empty:
            top_features_2 = self.model2_importance.head(5)
            
            headers = ["Feature", "Importance"]
            data = [[row['feature'], f"{row['importance']:.4f}"] for _, row in top_features_2.iterrows()]
            
            self.add_subsection_header("Top Features - Model 2")
            self.add_table(headers, data)
    
    def create_performance_comparison(self):
        """Create the performance comparison section."""
        self.add_section_header("3. Performance Comparison")
        
        perf_text = (
            "This section compares the predictive performance of both models. "
            "The metrics show how well each model predicts the November repayment rate, "
            "which is critical for accurate risk assessment."
        )
        
        self.add_paragraph(perf_text)
        
        # Add feature importance comparison plot
        feature_importance_path = os.path.join(self.data_dir, "feature_importance_comparison.png")
        self.add_image(feature_importance_path, caption="Feature Importance Comparison")
        
        # Add performance metrics comparison
        if self.comparison_data:
            perf_imp = self.comparison_data.get('performance_improvement', {})
            headers = ["Metric", "Model 1", "Model 2", "Improvement"]
            
            # Get average metrics from threshold dataframes
            model1_accuracy = self.model1_metrics['accuracy'].mean() if not self.model1_metrics.empty else 0
            model2_accuracy = self.model2_metrics['accuracy'].mean() if not self.model2_metrics.empty else 0
            
            model1_f1 = self.model1_metrics['f1_score'].mean() if not self.model1_metrics.empty else 0
            model2_f1 = self.model2_metrics['f1_score'].mean() if not self.model2_metrics.empty else 0
            
            model1_roi = self.model1_metrics['roi'].mean() if not self.model1_metrics.empty else 0
            model2_roi = self.model2_metrics['roi'].mean() if not self.model2_metrics.empty else 0
            
            data = [
                ["Accuracy", f"{model1_accuracy:.4f}", f"{model2_accuracy:.4f}", f"{perf_imp.get('accuracy', 0):.1f}%"],
                ["F1 Score", f"{model1_f1:.4f}", f"{model2_f1:.4f}", f"{perf_imp.get('f1', 0):.1f}%"],
                ["ROI", f"{model1_roi:.4f}", f"{model2_roi:.4f}", f"{perf_imp.get('roi', 0):.1f}%"]
            ]
            
            self.add_subsection_header("Performance Metrics")
            self.add_table(headers, data)
            
            # Interpretation
            rmse_imp = perf_imp.get('rmse', 0)
            r2_imp = perf_imp.get('r2', 0)
            
            if rmse_imp < 0 and r2_imp > 0:
                interpretation = (
                    "Model 2 shows significant improvement over Model 1 across all performance metrics. "
                    f"The {abs(rmse_imp):.1f}% reduction in RMSE and {abs(r2_imp):.1f}% increase in R² "
                    "demonstrate that September payment data provides valuable signal for predicting "
                    "November repayment rates."
                )
            elif rmse_imp < 0 and r2_imp < 0:
                interpretation = (
                    "Model 2 shows mixed results compared to Model 1. While there is a "
                    f"{abs(rmse_imp):.1f}% reduction in RMSE, the R² value decreased by {abs(r2_imp):.1f}%. "
                    "This suggests that while the average prediction error is smaller, the model's "
                    "explained variance has decreased."
                )
            elif rmse_imp > 0 and r2_imp > 0:
                interpretation = (
                    "Model 2 shows mixed results compared to Model 1. While the R² value increased by "
                    f"{abs(r2_imp):.1f}%, the RMSE increased by {abs(rmse_imp):.1f}%. "
                    "This suggests that while the model explains more variance, it may make larger "
                    "errors on some predictions."
                )
            else:
                interpretation = (
                    "Model 2 shows worse performance than Model 1 across metrics, which is unexpected. "
                    "This could indicate overfitting to the September payment data or issues with "
                    "the feature engineering process. Further investigation is recommended."
                )
                
            self.add_paragraph(interpretation)
    
    def create_profit_analysis(self):
        """Create the profit analysis section."""
        self.add_section_header("4. Profitability Analysis")
        
        profit_text = (
            "This section analyzes the business impact of each model in terms of profitability "
            "at different approval thresholds. The analysis considers both actual profit and "
            "return on investment (ROI) metrics."
        )
        
        self.add_paragraph(profit_text)
        
        # Add profit comparison plots
        profit_path = os.path.join(self.data_dir, "profit_comparison.png")
        self.add_image(profit_path, caption="Profit by Threshold Comparison")
        
        roi_path = os.path.join(self.data_dir, "roi_comparison.png")
        self.add_image(roi_path, caption="ROI by Threshold Comparison")
        
        # Add profit-roi tradeoff
        tradeoff_path = os.path.join(self.data_dir, "profit_roi_tradeoff.png")
        self.add_image(tradeoff_path, caption="Profit vs. ROI Tradeoff (with threshold values)")
        
        # Optimal thresholds
        if self.comparison_data:
            self.add_subsection_header("Optimal Thresholds")
            
            model1_opt = self.comparison_data.get('model1_optimal', {})
            model2_opt = self.comparison_data.get('model2_optimal', {})
            
            headers = ["Optimization Goal", "Model 1 Threshold", "Model 2 Threshold"]
            data = [
                ["Profit-focused", f"{model1_opt.get('profit', {}).get('threshold', 0):.2f}", 
                                 f"{model2_opt.get('profit', {}).get('threshold', 0):.2f}"],
                ["ROI-focused", f"{model1_opt.get('roi', {}).get('threshold', 0):.2f}", 
                               f"{model2_opt.get('roi', {}).get('threshold', 0):.2f}"]
            ]
            
            self.add_table(headers, data)
            
            # Business implications
            profit_imp = self.comparison_data.get('profit_improvement', 0)
            roi_imp = self.comparison_data.get('roi_improvement', 0)
            
            implications = (
                "The profitability analysis reveals that Model 2 (with September payment data) "
                f"{'improves' if profit_imp > 0 else 'reduces'} profit by {abs(profit_imp):.1f}% and "
                f"{'improves' if roi_imp > 0 else 'reduces'} ROI by {abs(roi_imp):.1f}% compared to Model 1 "
                "at their respective optimal thresholds.\n\n"
                
                "This suggests that incorporating September payment data allows for more accurate "
                "risk assessment, leading to better loan approval decisions that "
                f"{'maximize' if profit_imp > 0 and roi_imp > 0 else 'affect'} both profit and ROI."
            )
            
            self.add_paragraph(implications)
    
    def create_approval_rate_analysis(self):
        """Create the approval rate analysis section."""
        self.add_section_header("5. Approval Rate Analysis")
        
        approval_text = (
            "This section examines how different thresholds affect the loan approval rate, "
            "which has direct implications for business volume and risk management."
        )
        
        self.add_paragraph(approval_text)
        
        # Add approval rate plot
        approval_path = os.path.join(self.data_dir, "approval_rate_comparison.png")
        self.add_image(approval_path, caption="Approval Rate by Threshold")
        
        # Add F1 score plot to show classification performance
        f1_path = os.path.join(self.data_dir, "f1_comparison.png")
        self.add_image(f1_path, caption="F1 Score by Threshold (Binary Classification Performance)")
        
        # Analysis of approval rates at optimal thresholds
        if not self.model1_metrics.empty and not self.model2_metrics.empty and self.comparison_data:
            model1_opt = self.comparison_data.get('model1_optimal', {})
            model2_opt = self.comparison_data.get('model2_optimal', {})
            
            model1_profit_threshold = model1_opt.get('profit', {}).get('threshold', 0)
            model2_profit_threshold = model2_opt.get('profit', {}).get('threshold', 0)
            
            model1_roi_threshold = model1_opt.get('roi', {}).get('threshold', 0)
            model2_roi_threshold = model2_opt.get('roi', {}).get('threshold', 0)
            
            # Get approval rates at these thresholds
            model1_profit_approval = self.model1_metrics[
                self.model1_metrics['threshold'] == model1_profit_threshold
            ]['approval_rate'].values[0] if len(self.model1_metrics[
                self.model1_metrics['threshold'] == model1_profit_threshold
            ]) > 0 else 0
            
            model2_profit_approval = self.model2_metrics[
                self.model2_metrics['threshold'] == model2_profit_threshold
            ]['approval_rate'].values[0] if len(self.model2_metrics[
                self.model2_metrics['threshold'] == model2_profit_threshold
            ]) > 0 else 0
            
            model1_roi_approval = self.model1_metrics[
                self.model1_metrics['threshold'] == model1_roi_threshold
            ]['approval_rate'].values[0] if len(self.model1_metrics[
                self.model1_metrics['threshold'] == model1_roi_threshold
            ]) > 0 else 0
            
            model2_roi_approval = self.model2_metrics[
                self.model2_metrics['threshold'] == model2_roi_threshold
            ]['approval_rate'].values[0] if len(self.model2_metrics[
                self.model2_metrics['threshold'] == model2_roi_threshold
            ]) > 0 else 0
            
            headers = ["Optimization Goal", "Model 1 Approval Rate", "Model 2 Approval Rate", "Difference"]
            data = [
                ["Profit-focused", f"{model1_profit_approval:.1%}", f"{model2_profit_approval:.1%}", 
                 f"{(model2_profit_approval - model1_profit_approval) * 100:.1f}%"],
                ["ROI-focused", f"{model1_roi_approval:.1%}", f"{model2_roi_approval:.1%}", 
                 f"{(model2_roi_approval - model1_roi_approval) * 100:.1f}%"]
            ]
            
            self.add_subsection_header("Approval Rates at Optimal Thresholds")
            self.add_table(headers, data)
            
            # Business impact analysis
            approval_diff_profit = model2_profit_approval - model1_profit_approval
            approval_diff_roi = model2_roi_approval - model1_roi_approval
            
            impact_text = (
                "The analysis reveals that at profit-optimized thresholds, Model 2 would "
                f"{'increase' if approval_diff_profit > 0 else 'decrease'} the approval rate by "
                f"{abs(approval_diff_profit) * 100:.1f}% compared to Model 1. Similarly, at ROI-optimized "
                f"thresholds, Model 2 would {'increase' if approval_diff_roi > 0 else 'decrease'} the "
                f"approval rate by {abs(approval_diff_roi) * 100:.1f}%.\n\n"
                
                f"This {'higher' if approval_diff_profit > 0 and approval_diff_roi > 0 else 'different'} "
                "approval rate implies that incorporating September payment data allows for "
                f"{'more' if approval_diff_profit > 0 and approval_diff_roi > 0 else 'better'} "
                f"loan approvals while {'maintaining' if approval_diff_profit > 0 and approval_diff_roi > 0 else 'improving'} "
                "profit and ROI targets."
            )
            
            self.add_paragraph(impact_text)
    
    def create_recommendations(self):
        """Create the recommendations section."""
        self.add_section_header("6. Business Recommendations")
        
        if not self.comparison_data:
            self.add_paragraph("Error: Comparison data not available. Please run the model comparison first.")
            return
            
        # Extract key metrics for recommendations
        perf_imp = self.comparison_data.get('performance_improvement', {})
        profit_imp = self.comparison_data.get('profit_improvement', 0)
        roi_imp = self.comparison_data.get('roi_improvement', 0)
        
        model1_opt = self.comparison_data.get('model1_optimal', {})
        model2_opt = self.comparison_data.get('model2_optimal', {})
        
        model1_profit_threshold = model1_opt.get('profit', {}).get('threshold', 0)
        model2_profit_threshold = model2_opt.get('profit', {}).get('threshold', 0)
        
        model1_roi_threshold = model1_opt.get('roi', {}).get('threshold', 0)
        model2_roi_threshold = model2_opt.get('roi', {}).get('threshold', 0)
        
        # Implementation strategy
        self.add_subsection_header("Implementation Strategy")
        
        if profit_imp > 0 and roi_imp > 0:
            strategy = (
                "Based on the analysis, we recommend implementing a two-model approach:\n\n"
                
                "1. For new loan applications (no payment history):\n"
                f"   - Use Model 1 with a threshold of {model1_profit_threshold:.2f} for profit-focused decisions\n"
                f"   - Use Model 1 with a threshold of {model1_roi_threshold:.2f} for ROI-focused decisions\n\n"
                
                "2. For existing loans with payment history:\n"
                f"   - Use Model 2 with a threshold of {model2_profit_threshold:.2f} for profit-focused decisions\n"
                f"   - Use Model 2 with a threshold of {model2_roi_threshold:.2f} for ROI-focused decisions\n\n"
                
                "This dual approach allows for optimized decision-making at different stages of the loan lifecycle."
            )
        else:
            strategy = (
                "Based on the mixed results between Model 1 and Model 2, we recommend:\n\n"
                
                "1. For new loan applications (no payment history):\n"
                f"   - Use Model 1 with a threshold of {model1_profit_threshold:.2f} for profit-focused decisions\n"
                "   - Conduct further analysis to validate optimal thresholds for this model\n\n"
                
                "2. For existing loans with payment history:\n"
                "   - Perform additional validation on Model 2 to better understand its limitations\n"
                "   - Consider a hybrid approach that incorporates insights from both models\n\n"
                
                "Due to the mixed performance results, we recommend a phased implementation "
                "with careful monitoring of model performance in production."
            )
            
        self.add_paragraph(strategy)
        
        # Risk considerations
        self.add_subsection_header("Risk Considerations")
        
        risk_text = (
            "When implementing these models, consider the following risk factors:\n\n"
            
            "- Data Drift: The models should be monitored for performance degradation over time "
            "as economic conditions and customer behaviors change.\n\n"
            
            "- Model Interpretability: While the feature importance analysis provides insights "
            "into model decisions, additional explainability techniques may be needed for "
            "regulatory compliance and stakeholder understanding.\n\n"
            
            "- Generalization: The models should be validated on additional data sets to ensure "
            "they generalize well to new loans and different time periods.\n\n"
            
            "- Threshold Selection: The choice between profit-focused and ROI-focused thresholds "
            "should align with overall business strategy and risk appetite."
        )
        
        self.add_paragraph(risk_text)
    
    def generate_report(self):
        """Generate the complete model comparison report."""
        # Add title
        self.add_title("Loan Repayment Prediction Model Comparison", 
                       "Application-Time Features vs. Including September Payment Data")
        
        # Create report sections
        self.create_executive_summary()
        self.create_model_descriptions()
        self.create_performance_comparison()
        self.create_profit_analysis()
        self.create_approval_rate_analysis()
        self.create_recommendations()
        
        # Save the report
        report_path = os.path.join(self.output_dir, "model_comparison_report.pdf")
        self.pdf.output(report_path)
        
        print(f"Report generated successfully: {report_path}")
        
        return report_path


def create_markdown_report(
    data_dir: str = "data/processed/model_comparison",
    output_dir: str = "reports"
):
    """
    Create a Markdown version of the model comparison report.
    
    Args:
        data_dir: Directory containing model comparison results
        output_dir: Directory to save the report
    
    Returns:
        Path to the generated Markdown report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load comparison data
    try:
        with open(os.path.join(data_dir, "model_comparison.json"), 'r') as f:
            comparison_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: model_comparison.json not found in {data_dir}")
        comparison_data = {}
    
    # Extract key metrics
    perf_imp = comparison_data.get('performance_improvement', {})
    rmse_imp = perf_imp.get('rmse', 0)
    r2_imp = perf_imp.get('r2', 0)
    profit_imp = comparison_data.get('profit_improvement', 0)
    roi_imp = comparison_data.get('roi_improvement', 0)
    
    # Create Markdown content
    markdown_content = [
        "# Loan Repayment Prediction Model Comparison",
        "## Application-Time Features vs. Including September Payment Data",
        
        "## 1. Executive Summary",
        
        "This report compares two approaches for loan repayment rate prediction:",
        "- **Model 1** uses only application-time features",
        "- **Model 2** incorporates September payment data in addition to application-time features",
        "",
        f"Model 2 shows a {abs(rmse_imp):.1f}% {'decrease' if rmse_imp < 0 else 'increase'} in RMSE and a "
        f"{abs(r2_imp):.1f}% {'increase' if r2_imp > 0 else 'decrease'} in R² compared to Model 1. "
        f"This demonstrates the {'positive' if rmse_imp < 0 and r2_imp > 0 else 'mixed'} impact of "
        "including September payment data on predictive accuracy.",
        "",
        f"In terms of profitability, Model 2 shows a {abs(profit_imp):.1f}% {'increase' if profit_imp > 0 else 'decrease'} "
        f"in profit and a {abs(roi_imp):.1f}% {'increase' if roi_imp > 0 else 'decrease'} in ROI at their respective "
        "optimal thresholds. This highlights the business value of incorporating payment data when available.",
        
        "## 2. Model Descriptions",
        
        "### Model 1: Application-Time Features Only",
        
        "This model uses only features available at the time of loan application, "
        "making it suitable for new loan application decisions when no payment history exists. "
        "The features include:",
        "",
        "- Basic loan details (loan type, region, duka name, etc.)",
        "- Contract value and deposit amount",
        "- Historical metrics by region and territory",
        "- Temporal features (contract start date, day of week, etc.)",
        "- Geographic features (distances to key locations)",
        "",
        "This model represents a production-ready approach that can be used for all new loan applications.",
        
        "### Model 2: Including September Payment Data",
        
        "This model includes all application-time features plus data from the September payment period. "
        "The additional features include:",
        "",
        "- Cumulative amount paid up to September",
        "- September repayment rate",
        "- Payment velocity (payment per day)",
        "- Days between contract start and September",
        "",
        "This model can be used for ongoing risk assessment and portfolio management for existing loans "
        "where payment history is available. It provides better predictive performance but requires "
        "post-application data.",
        
        "## 3. Performance Comparison",
        
        "![Feature Importance Comparison](../data/processed/model_comparison/feature_importance_comparison.png)",
        "",
        "## 4. Profitability Analysis",
        
        "![Profit by Threshold Comparison](../data/processed/model_comparison/profit_comparison.png)",
        "",
        "![ROI by Threshold Comparison](../data/processed/model_comparison/roi_comparison.png)",
        "",
        "![Profit vs. ROI Tradeoff](../data/processed/model_comparison/profit_roi_tradeoff.png)",
        
        "## 5. Approval Rate Analysis",
        
        "![Approval Rate by Threshold](../data/processed/model_comparison/approval_rate_comparison.png)",
        "",
        "![F1 Score by Threshold](../data/processed/model_comparison/f1_comparison.png)",
        
        "## 6. Business Recommendations",
        
        "### Implementation Strategy",
    ]
    
    # Add recommendations based on performance
    if profit_imp > 0 and roi_imp > 0:
        markdown_content.extend([
            "Based on the analysis, we recommend implementing a two-model approach:",
            "",
            "1. For new loan applications (no payment history):",
            "   - Use Model 1 with an optimal threshold for profit-focused decisions",
            "   - Use a higher threshold for ROI-focused decisions",
            "",
            "2. For existing loans with payment history:",
            "   - Use Model 2 with its optimal threshold for better profitability and ROI",
            "",
            "This dual approach allows for optimized decision-making at different stages of the loan lifecycle."
        ])
    else:
        markdown_content.extend([
            "Based on the mixed results between Model 1 and Model 2, we recommend:",
            "",
            "1. For new loan applications (no payment history):",
            "   - Use Model 1 with its optimal threshold for profit-focused decisions",
            "   - Conduct further analysis to validate optimal thresholds for this model",
            "",
            "2. For existing loans with payment history:",
            "   - Perform additional validation on Model 2 to better understand its limitations",
            "   - Consider a hybrid approach that incorporates insights from both models",
            "",
            "Due to the mixed performance results, we recommend a phased implementation "
            "with careful monitoring of model performance in production."
        ])
    
    # Add risk considerations
    markdown_content.extend([
        "",
        "### Risk Considerations",
        "",
        "When implementing these models, consider the following risk factors:",
        "",
        "- **Data Drift**: The models should be monitored for performance degradation over time "
        "as economic conditions and customer behaviors change.",
        "",
        "- **Model Interpretability**: While the feature importance analysis provides insights "
        "into model decisions, additional explainability techniques may be needed for "
        "regulatory compliance and stakeholder understanding.",
        "",
        "- **Generalization**: The models should be validated on additional data sets to ensure "
        "they generalize well to new loans and different time periods.",
        "",
        "- **Threshold Selection**: The choice between profit-focused and ROI-focused thresholds "
        "should align with overall business strategy and risk appetite."
    ])
    
    # Write to file
    output_path = os.path.join(output_dir, "model_comparison_report.md")
    with open(output_path, 'w') as f:
        f.write('\n'.join(markdown_content))
    
    print(f"Markdown report generated successfully: {output_path}")
    return output_path
