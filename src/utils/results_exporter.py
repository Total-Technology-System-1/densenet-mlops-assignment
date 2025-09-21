"""
Results Export System - Day 2
Handles exporting benchmark results to various formats.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd


class ResultsExporter:
    """
    Handles exporting benchmark results to multiple formats.
    
    Supports:
    - CSV (required format for assignment)
    - JSON (detailed results)
    - Excel (for analysis)
    - Summary reports
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize results exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Required CSV columns as per assignment
        self.csv_columns = [
            'model_variant',
            'batch_size', 
            'device',
            'ram_usage_mb',
            'vram_usage_mb',
            'cpu_utilization_pct',
            'gpu_utilization_pct',
            'latency_ms',
            'throughput_samples_sec',
            'accuracy_top1',
            'accuracy_top5',
            'model_size_mb',
            'optimization_technique'
        ]
    
    def export_to_csv(
        self, 
        results: List[Dict[str, Any]], 
        output_path: Union[str, Path]
    ) -> str:
        """
        Export results to CSV format (assignment requirement).
        
        Args:
            results: List of benchmark results
            output_path: Path to output CSV file
            
        Returns:
            Path to created CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
                writer.writeheader()
                
                for result in results:
                    # Filter and map result to required columns
                    csv_row = {}
                    for column in self.csv_columns:
                        if column in result:
                            csv_row[column] = result[column]
                        else:
                            # Handle missing columns with appropriate defaults
                            csv_row[column] = self._get_default_value(column, result)
                    
                    writer.writerow(csv_row)
            
            self.logger.info(f"CSV export completed: {output_path}")
            self.logger.info(f"Exported {len(results)} benchmark results")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            raise
    
    def _get_default_value(self, column: str, result: Dict[str, Any]) -> Any:
        """Get default value for missing columns."""
        defaults = {
            'model_variant': f"densenet121_{result.get('optimization_technique', 'unknown')}",
            'batch_size': result.get('batch_size', 0),
            'device': result.get('device', 'unknown'),
            'ram_usage_mb': result.get('ram_usage_mb', 0),
            'vram_usage_mb': result.get('vram_usage_mb', 0),
            'cpu_utilization_pct': result.get('cpu_utilization_pct', 0),
            'gpu_utilization_pct': result.get('gpu_utilization_pct', 0),
            'latency_ms': result.get('mean_latency_ms', result.get('latency_ms', 0)),
            'throughput_samples_sec': result.get('throughput_samples_sec', 0),
            'accuracy_top1': result.get('accuracy_top1'),
            'accuracy_top5': result.get('accuracy_top5'),
            'model_size_mb': result.get('model_size_mb', 0),
            'optimization_technique': result.get('optimization_technique', 'unknown')
        }
        
        return defaults.get(column, '')
    
    def export_to_json(
        self, 
        results: List[Dict[str, Any]], 
        output_path: Union[str, Path],
        include_metadata: bool = True
    ) -> str:
        """
        Export results to JSON format with full details.
        
        Args:
            results: List of benchmark results
            output_path: Path to output JSON file
            include_metadata: Whether to include metadata
            
        Returns:
            Path to created JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            export_data = {
                'results': results
            }
            
            if include_metadata:
                export_data['metadata'] = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_results': len(results),
                    'configuration': self.config,
                    'export_version': '1.0'
                }
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, default=str)
            
            self.logger.info(f"JSON export completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            raise
    
    def export_to_excel(
        self, 
        results: List[Dict[str, Any]], 
        output_path: Union[str, Path]
    ) -> str:
        """
        Export results to Excel format with multiple sheets.
        
        Args:
            results: List of benchmark results
            output_path: Path to output Excel file
            
        Returns:
            Path to created Excel file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main results sheet
                df = pd.DataFrame(results)
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Summary sheet
                summary = self._generate_summary_stats(results)
                summary_df = pd.DataFrame(summary.items(), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Performance comparison sheet
                if len(results) > 1:
                    comparison = self._generate_comparison_table(results)
                    comparison.to_excel(writer, sheet_name='Comparison', index=False)
            
            self.logger.info(f"Excel export completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export Excel: {e}")
            raise
    
    def _generate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not results:
            return {}
        
        # Filter successful results
        successful = [r for r in results if 'error' not in r and 'mean_latency_ms' in r]
        
        if not successful:
            return {'error': 'No successful results to summarize'}
        
        # Calculate statistics
        latencies = [r['mean_latency_ms'] for r in successful]
        throughputs = [r['throughput_samples_sec'] for r in successful]
        memory_usage = [r.get('ram_usage_mb', 0) for r in successful]
        
        import statistics
        
        return {
            'total_benchmarks': len(results),
            'successful_benchmarks': len(successful),
            'failed_benchmarks': len(results) - len(successful),
            'avg_latency_ms': statistics.mean(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'avg_throughput': statistics.mean(throughputs),
            'max_throughput': max(throughputs),
            'avg_memory_mb': statistics.mean(memory_usage),
            'techniques_tested': len(set(r.get('optimization_technique', 'unknown') for r in successful))
        }
    
    def _generate_comparison_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate comparison table across techniques and batch sizes."""
        try:
            df = pd.DataFrame(results)
            
            # Pivot table for comparison
            comparison = df.pivot_table(
                index='batch_size',
                columns='optimization_technique', 
                values=['mean_latency_ms', 'throughput_samples_sec', 'ram_usage_mb'],
                aggfunc='mean'
            )
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"Could not generate comparison table: {e}")
            return pd.DataFrame()
    
    def generate_summary_report(
        self, 
        results: List[Dict[str, Any]], 
        output_path: Union[str, Path]
    ) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            results: List of benchmark results
            output_path: Path to output report file
            
        Returns:
            Path to created report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            summary_stats = self._generate_summary_stats(results)
            
            # Generate report content
            report = self._create_markdown_report(results, summary_stats)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Summary report generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            raise
    
    def _create_markdown_report(
        self, 
        results: List[Dict[str, Any]], 
        summary: Dict[str, Any]
    ) -> str:
        """Create markdown formatted report."""
        
        report = f"""# DenseNet MLOps Benchmark Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Benchmarks**: {summary.get('total_benchmarks', 0)}
- **Successful**: {summary.get('successful_benchmarks', 0)}
- **Failed**: {summary.get('failed_benchmarks', 0)}
- **Optimization Techniques Tested**: {summary.get('techniques_tested', 0)}

## Performance Summary

### Latency
- **Average**: {summary.get('avg_latency_ms', 0):.3f} ms
- **Best**: {summary.get('min_latency_ms', 0):.3f} ms
- **Worst**: {summary.get('max_latency_ms', 0):.3f} ms

### Throughput
- **Average**: {summary.get('avg_throughput', 0):.2f} samples/sec
- **Best**: {summary.get('max_throughput', 0):.2f} samples/sec

### Memory Usage
- **Average RAM**: {summary.get('avg_memory_mb', 0):.1f} MB

## Detailed Results

"""
        
        # Add detailed results table
        if results:
            successful_results = [r for r in results if 'error' not in r]
            
            if successful_results:
                report += "| Technique | Batch Size | Latency (ms) | Throughput | RAM (MB) |\n"
                report += "|-----------|------------|--------------|------------|----------|\n"
                
                for result in successful_results:
                    technique = result.get('optimization_technique', 'unknown')
                    batch_size = result.get('batch_size', 0)
                    latency = result.get('mean_latency_ms', 0)
                    throughput = result.get('throughput_samples_sec', 0)
                    ram = result.get('ram_usage_mb', 0)
                    
                    report += f"| {technique} | {batch_size} | {latency:.3f} | {throughput:.2f} | {ram:.1f} |\n"
        
        # Add failures if any
        failed_results = [r for r in results if 'error' in r]
        if failed_results:
            report += "\n## Failed Benchmarks\n\n"
            for result in failed_results:
                technique = result.get('optimization_technique', 'unknown')
                batch_size = result.get('batch_size', 0)
                error = result.get('error', 'Unknown error')
                report += f"- **{technique}** (batch {batch_size}): {error}\n"
        
        # Add recommendations
        report += "\n## Recommendations\n\n"
        if successful_results:
            # Find best performing configuration
            best_latency = min(successful_results, key=lambda x: x.get('mean_latency_ms', float('inf')))
            best_throughput = max(successful_results, key=lambda x: x.get('throughput_samples_sec', 0))
            
            report += f"- **Best Latency**: {best_latency.get('optimization_technique')} with batch size {best_latency.get('batch_size')} ({best_latency.get('mean_latency_ms', 0):.3f} ms)\n"
            report += f"- **Best Throughput**: {best_throughput.get('optimization_technique')} with batch size {best_throughput.get('batch_size')} ({best_throughput.get('throughput_samples_sec', 0):.2f} samples/sec)\n"
        
        report += "\n---\n*Generated by DenseNet MLOps Benchmarking Suite*\n"
        
        return report
    
    def export_all_formats(
        self, 
        results: List[Dict[str, Any]], 
        base_path: Union[str, Path],
        filename_prefix: str = "benchmark_results"
    ) -> Dict[str, str]:
        """
        Export results to all supported formats.
        
        Args:
            results: List of benchmark results
            base_path: Base directory for outputs
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping format names to file paths
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        try:
            # CSV (required)
            csv_path = self.export_to_csv(results, base_path / f"{filename_prefix}.csv")
            exported_files['csv'] = csv_path
            
            # JSON (detailed)
            json_path = self.export_to_json(results, base_path / f"{filename_prefix}.json")
            exported_files['json'] = json_path
            
            # Excel (analysis)
            try:
                excel_path = self.export_to_excel(results, base_path / f"{filename_prefix}.xlsx")
                exported_files['excel'] = excel_path
            except ImportError:
                self.logger.warning("pandas/openpyxl not available, skipping Excel export")
            
            # Summary report
            report_path = self.generate_summary_report(results, base_path / f"{filename_prefix}_report.md")
            exported_files['report'] = report_path
            
            self.logger.info(f"Results exported to {len(exported_files)} formats")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Failed to export all formats: {e}")
            raise


def export_results(
    results: List[Dict[str, Any]], 
    output_dir: str = "./results",
    formats: List[str] = None
) -> Dict[str, str]:
    """
    Convenience function to export results.
    
    Args:
        results: Benchmark results to export
        output_dir: Output directory
        formats: List of formats to export ('csv', 'json', 'excel', 'report')
        
    Returns:
        Dictionary mapping format names to file paths
    """
    if formats is None:
        formats = ['csv', 'json', 'report']
    
    exporter = ResultsExporter()
    
    exported = {}
    output_path = Path(output_dir)
    
    if 'csv' in formats:
        exported['csv'] = exporter.export_to_csv(results, output_path / "benchmark_results.csv")
    
    if 'json' in formats:
        exported['json'] = exporter.export_to_json(results, output_path / "benchmark_results.json")
    
    if 'excel' in formats:
        try:
            exported['excel'] = exporter.export_to_excel(results, output_path / "benchmark_results.xlsx")
        except ImportError:
            pass
    
    if 'report' in formats:
        exported['report'] = exporter.generate_summary_report(results, output_path / "benchmark_report.md")
    
    return exported


if __name__ == "__main__":
    # Test with sample data
    sample_results = [
        {
            'model_variant': 'densenet121_baseline',
            'batch_size': 4,
            'device': 'cuda',
            'mean_latency_ms': 15.2,
            'throughput_samples_sec': 263.2,
            'ram_usage_mb': 1024.5,
            'vram_usage_mb': 512.3,
            'cpu_utilization_pct': 45.2,
            'gpu_utilization_pct': 78.1,
            'model_size_mb': 28.7,
            'optimization_technique': 'baseline'
        }
    ]
    
    exported = export_results(sample_results, "./test_results")
    print(f"Test export completed: {exported}")