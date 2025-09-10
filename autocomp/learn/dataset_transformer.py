#!/usr/bin/env python3
"""
Script to transform rerun_results.json from rerun_code.py into a dataset format 
suitable for LLM fine-tuning, similar to HuggingFaceH4/Multilingual-Thinking.

The output dataset has columns: ['reasoning_language', 'developer', 'user', 'analysis', 'final', 'messages']
Each entry's messages contain the parent_code, plan, and performance change information.

Performance Classification:
The script includes a configurable performance level classification system with the following default levels:
- significantly_improved: >= 10% improvement
- moderately_improved: >= 5% improvement  
- slightly_improved: >= 1% improvement
- maintained: >= -1% improvement
- slightly_degraded: >= -5% improvement
- moderately_degraded: >= -10% improvement
- significantly_degraded: < -10% improvement

You can customize these levels using the update_performance_levels() method.

Incorrect Sample Filtering:
The script supports filtering incorrect samples using the --keep-incorrect-probability parameter.
By default (0.0), all incorrect samples are filtered out. Setting this to 1.0 keeps all incorrect
samples. Values between 0.0 and 1.0 randomly keep that fraction of incorrect samples, which can
be useful for training models on both positive and negative examples.
"""

import json
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetTransformer:
    def __init__(self, input_file: str, output_file: str = "fine_tuning_dataset.json", 
                 keep_incorrect_probability: float = 0.0):
        """
        Initialize the dataset transformer.
        
        Args:
            input_file: Path to the rerun_results.json file
            output_file: Path for the output dataset file
            keep_incorrect_probability: Probability (0.0-1.0) of keeping incorrect samples.
                                      0.0 means no incorrect samples are kept,
                                      1.0 means all incorrect samples are kept.
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.keep_incorrect_probability = max(0.0, min(1.0, keep_incorrect_probability))
        self.data = []
        self.transformed_data = []
        self.filtered_stats = {'total_processed': 0, 'incorrect_filtered': 0, 'incorrect_kept': 0}
        
        # Configure performance improvement levels
        self.performance_levels = {
            'significantly_improved': {'threshold': 25.0, 'description': 'significantly improved'},
            'moderately_improved': {'threshold': 5.0, 'description': 'moderately improved'},
            'slightly_improved': {'threshold': 1.0, 'description': 'slightly improved'},
            'maintained': {'threshold': -1.0, 'description': 'maintained'},
            'slightly_degraded': {'threshold': -5.0, 'description': 'slightly degraded'},
            'moderately_degraded': {'threshold': -10.0, 'description': 'moderately degraded'},
            'significantly_degraded': {'threshold': float('-inf'), 'description': 'significantly degraded'}
        }
    
    def load_rerun_results(self):
        """Load the rerun results from JSON or pickle file."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file {self.input_file} not found")
        
        if self.input_file.suffix == '.json':
            with open(self.input_file, 'r') as f:
                self.data = json.load(f)
        elif self.input_file.suffix == '.pkl':
            with open(self.input_file, 'rb') as f:
                # If it's a list of dataclass objects, convert to dicts
                raw_data = pickle.load(f)
                if hasattr(raw_data[0], '__dataclass_fields__'):
                    from dataclasses import asdict
                    self.data = [asdict(item) for item in raw_data]
                else:
                    self.data = raw_data
        else:
            raise ValueError(f"Unsupported file format: {self.input_file.suffix}")
        
        logger.info(f"Loaded {len(self.data)} entries from {self.input_file}")
    
    def compute_performance_change(self, performance_metrics: Dict[str, Any], 
                                 parent_performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute performance change between implementation and parent code.
        
        Returns:
            Dict containing performance change information
        """
        result = {
            'is_correct': performance_metrics.get('correct', False),
            'performance_change': None,
            'performance_improvement': None,
            'metric_type': None,
            'parent_score': None,
            'impl_score': None,
            'error_message': None
        }
        
        if not result['is_correct']:
            result['error_message'] = performance_metrics.get('error', 'Implementation is incorrect')
            return result
        
        # Check for latency (Gemmini backend)
        if 'latency' in performance_metrics and 'latency' in parent_performance_metrics:
            parent_latency = parent_performance_metrics['latency']
            impl_latency = performance_metrics['latency']
            
            if parent_latency != float('inf') and impl_latency != float('inf'):
                result['metric_type'] = 'latency'
                result['parent_score'] = parent_latency
                result['impl_score'] = impl_latency
                result['performance_change'] = impl_latency - parent_latency
                result['performance_improvement'] = (parent_latency - impl_latency) / parent_latency * 100
        
        # Check for runtime (CUDA backend)
        elif 'runtime' in performance_metrics and 'runtime' in parent_performance_metrics:
            parent_runtime = parent_performance_metrics['runtime']
            impl_runtime = performance_metrics['runtime']
            
            if parent_runtime != float('inf') and impl_runtime != float('inf'):
                result['metric_type'] = 'runtime'
                result['parent_score'] = parent_runtime
                result['impl_score'] = impl_runtime
                result['performance_change'] = impl_runtime - parent_runtime
                result['performance_improvement'] = (parent_runtime - impl_runtime) / parent_runtime * 100
        
        return result
    
    def classify_performance_improvement(self, performance_improvement: Optional[float]) -> str:
        """
        Classify performance improvement into predefined levels.
        
        Args:
            performance_improvement: Performance improvement percentage (positive = better)
            
        Returns:
            String description of the performance level
        """
        if performance_improvement is None:
            return 'maintained'  # Default for when we can't measure
        
        # Sort levels by threshold in descending order to find the first match
        sorted_levels = sorted(self.performance_levels.items(), 
                             key=lambda x: x[1]['threshold'], reverse=True)
        
        for level_name, level_info in sorted_levels:
            if performance_improvement >= level_info['threshold']:
                return level_info['description']
        
        # Fallback (should not reach here due to -inf threshold)
        return 'significantly degraded'
    
    def update_performance_levels(self, custom_levels: Dict[str, Dict[str, Any]]):
        """
        Update the performance level configuration.
        
        Args:
            custom_levels: Dictionary with performance level configuration.
                          Each key should map to a dict with 'threshold' and 'description' keys.
                          
        Example:
            transformer.update_performance_levels({
                'greatly_improved': {'threshold': 20.0, 'description': 'greatly improved'},
                'improved': {'threshold': 5.0, 'description': 'improved'},
                'maintained': {'threshold': -5.0, 'description': 'maintained'},
                'degraded': {'threshold': float('-inf'), 'description': 'degraded'}
            })
        """
        self.performance_levels = custom_levels
    
    def create_user_content(self, entry: Dict[str, Any]) -> str:
        """
        Create the unified user content used in both 'user' field and messages.
        
        Args:
            entry: Original rerun result entry
            
        Returns:
            Unified user content string
        """
        return f"""We want to optimize this Gemmini accelerator code:

Original code:
```{entry.get('prob_type', 'code')}
{entry.get('parent_code', '')}
```

Optimization plan:
{entry.get('plan', '')}

Please analyze the expected performance impact of implementing this optimization plan."""
    
    def create_analysis(self, entry: Dict[str, Any], perf_info: Dict[str, Any]) -> str:
        """
        Create the analysis section with generated code and performance reasoning.
        
        Args:
            entry: Original rerun result entry
            perf_info: Performance change information
            
        Returns:
            Analysis string containing generated code and performance reasoning
        """
        analysis_parts = []
        
        # Include the generated implementation code
        impl_code = entry.get('implementation_code', '')
        if impl_code:
            analysis_parts.append(f"Generated implementation:\n```{entry.get('prob_type', 'code')}\n{impl_code}\n```\n")
        
        # Performance analysis
        if perf_info['is_correct']:
            if perf_info['metric_type']:
                performance_level = self.classify_performance_improvement(perf_info['performance_improvement'])
                improvement_pct = perf_info['performance_improvement']
                
                if improvement_pct > 0:
                    analysis_parts.append(f"The optimization {performance_level} performance by {improvement_pct:.2f}%, reducing {perf_info['metric_type']} from {perf_info['parent_score']:.0f} to {perf_info['impl_score']:.0f}.")
                elif improvement_pct < 0:
                    analysis_parts.append(f"The optimization {performance_level} performance by {abs(improvement_pct):.2f}%, increasing {perf_info['metric_type']} from {perf_info['parent_score']:.0f} to {perf_info['impl_score']:.0f}.")
                else:
                    analysis_parts.append(f"The optimization {performance_level} the same performance with {perf_info['metric_type']} remaining at {perf_info['parent_score']:.0f}.")
            else:
                analysis_parts.append("The optimization produced a functionally correct implementation, though detailed performance metrics are not available.")
        else:
            analysis_parts.append("The optimization failed to produce a correct implementation, likely due to bugs introduced during the transformation.")
        
        # Add utilization information if available
        parent_perf = entry.get('parent_performance_metrics', {})
        impl_perf = entry.get('performance_metrics', {})
        
        if 'spad_util' in parent_perf and 'acc_util' in parent_perf:
            parent_spad = parent_perf['spad_util'] * 100
            parent_acc = parent_perf['acc_util'] * 100
            analysis_parts.append(f"Original scratchpad utilization: {parent_spad:.1f}%, accumulator utilization: {parent_acc:.1f}%.")
            
            if perf_info['is_correct'] and 'spad_util' in impl_perf and 'acc_util' in impl_perf:
                impl_spad = impl_perf['spad_util'] * 100
                impl_acc = impl_perf['acc_util'] * 100
                analysis_parts.append(f"Optimized scratchpad utilization: {impl_spad:.1f}%, accumulator utilization: {impl_acc:.1f}%.")
        
        # Add spad_acc_stats if available
        if 'spad_acc_stats' in parent_perf and parent_perf['spad_acc_stats']:
            stats_text = ' '.join(parent_perf['spad_acc_stats'])
            analysis_parts.append(f"Utilization feedback: {stats_text}")
        
        return '\n'.join(analysis_parts)
    
    def create_assistant_content(self, entry: Dict[str, Any], perf_info: Dict[str, Any]) -> str:
        """
        Create the assistant message content that will be used in the 'final' field.
        
        Args:
            entry: Original rerun result entry
            perf_info: Performance change information
            
        Returns:
            Assistant message content string
        """
        impl_code = entry.get('implementation_code', '')
        
        if perf_info['is_correct']:
            if perf_info['metric_type']:
                performance_text = f"""**Performance Analysis:**
- Implementation Status: ✅ Correct
- Metric: {perf_info['metric_type']}
- Original Score: {perf_info['parent_score']:.0f}
- Optimized Score: {perf_info['impl_score']:.0f}
- Performance Change: {perf_info['performance_change']:.0f}
- Performance Improvement: {perf_info['performance_improvement']:.2f}%

The optimization {self.classify_performance_improvement(perf_info['performance_improvement'])} the performance."""
            else:
                performance_text = f"""**Performance Analysis:**
- Implementation Status: ✅ Correct
- Performance metrics are not available for detailed comparison, but the implementation passes all correctness tests."""
        else:
            performance_text = f"""The optimization attempt resulted in an incorrect implementation.

**Analysis:**
- Implementation Status: ❌ Incorrect
- The optimization plan introduced bugs or logical errors that caused the implementation to fail correctness tests."""
        
        return performance_text
    
    def create_messages(self, entry: Dict[str, Any], perf_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create the messages list for the dataset entry.
        
        Args:
            entry: Original rerun result entry
            perf_info: Performance change information
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # System message explaining the task
        system_message = {
            "content": "You are an expert programmer of the Gemmini deep learning accelerator.",
            "role": "system",
            "thinking": None
        }
        messages.append(system_message)
        
        # User message with parent code and plan
        user_content = self.create_user_content(entry)
        
        user_message = {
            "content": user_content,
            "role": "user", 
            "thinking": None
        }
        messages.append(user_message)
        
        # Assistant response with implementation and analysis
        performance_text = self.create_assistant_content(entry, perf_info)
        
        # Create analysis for thinking field
        analysis_content = self.create_analysis(entry, perf_info)
        
        assistant_message = {
            "content": performance_text,
            "role": "assistant",
            "thinking": analysis_content
        }
        messages.append(assistant_message)
        
        return messages
    
    def transform_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single rerun result entry into dataset format.
        
        Args:
            entry: Single entry from rerun results
            
        Returns:
            Transformed entry in dataset format
        """
        # Compute performance change
        perf_info = self.compute_performance_change(
            entry.get('performance_metrics', {}),
            entry.get('parent_performance_metrics', {})
        )
        
        # Create messages
        messages = self.create_messages(entry, perf_info)
        
        # Create analysis with performance reasoning
        analysis = self.create_analysis(entry, perf_info)
        
        # User request with old code and plan
        user_request = self.create_user_content(entry)
        
        # Get the assistant message content for the 'final' field
        assistant_content = self.create_assistant_content(entry, perf_info)
        
        # Create the transformed entry
        transformed_entry = {
            'reasoning_language': 'English',
            'developer': 'You are an expert programmer of the Gemmini deep learning accelerator.',
            'user': user_request,
            'analysis': analysis,
            'final': assistant_content,
            'messages': messages,
            # Additional metadata for filtering/analysis
            'metadata': {
                'experiment_name': entry.get('experiment_name', ''),
                'iteration': entry.get('iteration', 0),
                'prob_type': entry.get('prob_type', ''),
                'prob_id': entry.get('prob_id', ''),
                'simulator': entry.get('simulator', ''),
                'backend_type': entry.get('backend_type', ''),
                'file_path': entry.get('file_path', ''),
                'performance_info': perf_info
            }
        }
        
        return transformed_entry
    
    def transform_dataset(self):
        """Transform all entries in the dataset, filtering incorrect samples based on probability."""
        logger.info("Starting dataset transformation...")
        logger.info(f"Keep incorrect probability: {self.keep_incorrect_probability}")
        
        for i, entry in enumerate(self.data):
            try:
                transformed_entry = self.transform_entry(entry)
                self.filtered_stats['total_processed'] += 1
                
                # Check if this is an incorrect sample
                is_correct = transformed_entry['metadata']['performance_info']['is_correct']
                
                if is_correct:
                    # Always keep correct samples
                    self.transformed_data.append(transformed_entry)
                else:
                    # For incorrect samples, keep based on probability
                    if random.random() < self.keep_incorrect_probability:
                        self.transformed_data.append(transformed_entry)
                        self.filtered_stats['incorrect_kept'] += 1
                    else:
                        self.filtered_stats['incorrect_filtered'] += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.data)} entries")
                    
            except Exception as e:
                logger.error(f"Error transforming entry {i}: {e}")
                continue
        
        logger.info(f"Transformation complete. {len(self.transformed_data)} entries kept out of {self.filtered_stats['total_processed']} processed.")
        if self.filtered_stats['incorrect_filtered'] > 0 or self.filtered_stats['incorrect_kept'] > 0:
            logger.info(f"Incorrect samples: {self.filtered_stats['incorrect_kept']} kept, {self.filtered_stats['incorrect_filtered']} filtered")
    
    def save_dataset(self):
        """Save the transformed dataset."""
        # Save as JSON
        with open(self.output_file, 'w') as f:
            json.dump(self.transformed_data, f, indent=2)
        
        # Also save as CSV for easy inspection (if pandas is available)
        if HAS_PANDAS:
            csv_file = self.output_file.with_suffix('.csv')
            
            # Flatten for CSV (exclude complex nested structures)
            csv_data = []
            for entry in self.transformed_data:
                csv_entry = {
                    'reasoning_language': entry['reasoning_language'],
                    'developer': entry['developer'],
                    'user': entry['user'],
                    'analysis': entry['analysis'],
                    'final': entry['final'],
                    'experiment_name': entry['metadata']['experiment_name'],
                    'iteration': entry['metadata']['iteration'],
                    'prob_type': entry['metadata']['prob_type'],
                    'prob_id': entry['metadata']['prob_id'],
                    'is_correct': entry['metadata']['performance_info']['is_correct'],
                    'performance_improvement': entry['metadata']['performance_info']['performance_improvement'],
                    'metric_type': entry['metadata']['performance_info']['metric_type']
                }
                csv_data.append(csv_entry)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False)
            logger.info(f"CSV summary saved to {csv_file}")
        else:
            logger.info("pandas not available, skipping CSV export")
        
        logger.info(f"Dataset saved to {self.output_file}")
    
    def print_statistics(self):
        """Print statistics about the transformed dataset."""
        if not self.transformed_data:
            logger.info("No data to analyze")
            return
        
        total_entries = len(self.transformed_data)
        correct_entries = sum(1 for entry in self.transformed_data 
                            if entry['metadata']['performance_info']['is_correct'])
        
        # Performance improvement statistics
        improvements = []
        for entry in self.transformed_data:
            perf_info = entry['metadata']['performance_info']
            if perf_info['is_correct'] and perf_info['performance_improvement'] is not None:
                improvements.append(perf_info['performance_improvement'])
        
        # Problem type distribution
        prob_types = {}
        for entry in self.transformed_data:
            prob_type = entry['metadata']['prob_type']
            prob_types[prob_type] = prob_types.get(prob_type, 0) + 1
        
        logger.info(f"\n=== DATASET STATISTICS ===")
        logger.info(f"Total entries in final dataset: {total_entries}")
        logger.info(f"Correct implementations: {correct_entries}")
        logger.info(f"Incorrect implementations: {total_entries - correct_entries}")
        logger.info(f"Success rate: {correct_entries/total_entries*100:.1f}%")
        
        # Show filtering statistics
        if self.filtered_stats['total_processed'] > 0:
            logger.info(f"\n=== FILTERING STATISTICS ===")
            logger.info(f"Total entries processed: {self.filtered_stats['total_processed']}")
            logger.info(f"Entries kept in final dataset: {total_entries}")
            logger.info(f"Keep incorrect probability: {self.keep_incorrect_probability}")
            if self.filtered_stats['incorrect_filtered'] > 0 or self.filtered_stats['incorrect_kept'] > 0:
                total_incorrect = self.filtered_stats['incorrect_filtered'] + self.filtered_stats['incorrect_kept']
                logger.info(f"Total incorrect samples processed: {total_incorrect}")
                logger.info(f"Incorrect samples kept: {self.filtered_stats['incorrect_kept']} ({self.filtered_stats['incorrect_kept']/total_incorrect*100:.1f}%)")
                logger.info(f"Incorrect samples filtered out: {self.filtered_stats['incorrect_filtered']} ({self.filtered_stats['incorrect_filtered']/total_incorrect*100:.1f}%)")
        
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            positive_improvements = sum(1 for x in improvements if x > 0)
            logger.info(f"\n=== PERFORMANCE STATISTICS ===")
            logger.info(f"Entries with performance metrics: {len(improvements)}")
            logger.info(f"Average performance improvement: {avg_improvement:.2f}%")
            logger.info(f"Entries with positive improvement: {positive_improvements}/{len(improvements)} ({positive_improvements/len(improvements)*100:.1f}%)")
        
        logger.info(f"\n=== PROBLEM TYPE DISTRIBUTION ===")
        for prob_type, count in sorted(prob_types.items()):
            logger.info(f"  {prob_type}: {count} ({count/total_entries*100:.1f}%)")
    
    def run(self):
        """Run the complete transformation pipeline."""
        self.load_rerun_results()
        self.transform_dataset()
        self.save_dataset()
        self.print_statistics()


def main():
    """Main function to run the dataset transformation."""
    parser = argparse.ArgumentParser(description="Transform rerun results into fine-tuning dataset format")
    parser.add_argument("input_file", help="Path to rerun_results.json or rerun_results.pkl file")
    parser.add_argument("--output-file", default="fine_tuning_dataset.json",
                       help="Output file path (default: fine_tuning_dataset.json)")
    parser.add_argument("--keep-incorrect-probability", type=float, default=0.0,
                       help="Probability (0.0-1.0) of keeping incorrect samples in the dataset. "
                            "0.0 means no incorrect samples are kept (default), "
                            "1.0 means all incorrect samples are kept.")
    
    args = parser.parse_args()
    
    # Validate keep_incorrect_probability
    if not (0.0 <= args.keep_incorrect_probability <= 1.0):
        parser.error("--keep-incorrect-probability must be between 0.0 and 1.0")
    
    # Create and run the transformer
    transformer = DatasetTransformer(args.input_file, args.output_file, args.keep_incorrect_probability)
    transformer.run()


if __name__ == "__main__":
    main()
