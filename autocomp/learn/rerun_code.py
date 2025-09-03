 #!/usr/bin/env python3
"""
Script to rerun all generated code iterations and save performance data.

This script:
1. Finds all generated-code-iter-x directories under output/ and output_old/
2. For each iteration, extracts the code before (parent), plan, and code after (implementation)
3. Reruns the code using the evaluate_code function
4. Saves the performance results along with the corresponding data
"""

import os
import pathlib
import glob
import json
import pickle
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

# Import the necessary classes from the autocomp package
from autocomp.backend.gemmini_eval import GemminiHardwareBackend
from autocomp.backend.kb_eval import KBHardwareBackend
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeIterationData:
    """Data structure to hold information about a code iteration."""
    experiment_name: str
    iteration: int
    parent_code: str
    plan: str
    implementation_code: str
    performance_metrics: Dict[str, Any]
    parent_performance_metrics: Dict[str, Any]  # Performance metrics for the parent code
    file_path: str
    prob_type: str
    prob_id: str
    simulator: str
    backend_type: str

@dataclass
class ImplementationInfo:
    """Helper class to hold implementation file information."""
    file_path: pathlib.Path
    code: str
    parent_code: str
    parent_candidate: Optional[CodeCandidate]
    plan: str
    candidate_idx: int
    plan_idx: int
    model: str

class CodeRerunner:
    def __init__(self, output_dirs: List[str] = None, experiment_filter: str = None):
        """
        Initialize the CodeRerunner.
        
        Args:
            output_dirs: List of output directories to scan. Defaults to ['output', 'output_old']
            experiment_filter: Filter string that must be present in experiment directory names. 
                             If None, no filtering is applied.
        """
        if output_dirs is None:
            output_dirs = ['output', 'output_old']
        
        self.output_dirs = [pathlib.Path(d) for d in output_dirs]
        self.experiment_filter = experiment_filter
        self.batch_size = 300  # Number of implementations to evaluate in each batch
        self.results = []
        
        # Create intermediate results directory
        self.intermediate_dir = pathlib.Path(f"rerun_intermediate")
        self.intermediate_dir.mkdir(exist_ok=True)
        logger.info(f"Created intermediate results directory: {self.intermediate_dir}")
    
    def extract_experiment_info(self, experiment_path: pathlib.Path) -> Dict[str, str]:
        """Extract experiment information from the directory name."""
        name = experiment_path.name
        
        # Parse experiment name to extract key information
        # Format: prob_id_beam_iters{N}_{simulator}_{model}_...
        parts = name.split('_')
        
        info = {
            'experiment_name': name,
            'prob_type': parts[0] if len(parts) > 0 else 'unknown',
            'prob_id': parts[1] if len(parts) > 1 else '0',
            'simulator': 'unknown',
            'backend_type': 'unknown'
        }
        
        # Extract simulator info
        if 'firesim' in name:
            info['simulator'] = 'firesim'
            info['backend_type'] = 'gemmini'
        elif 'spike' in name:
            info['simulator'] = 'spike'
            info['backend_type'] = 'gemmini'
        elif 'kernelbench' in name:
            info['simulator'] = 'kernelbench'
            info['backend_type'] = 'cuda'
        else:
            # Try to infer from problem type
            if any(x in info['prob_type'] for x in ['exo', 'admm', 'gpt']):
                info['backend_type'] = 'gemmini'
                info['simulator'] = 'spike'  # Default for gemmini
            else:
                info['backend_type'] = 'cuda'
                info['simulator'] = 'kernelbench'  # Default for cuda
        
        return info
    
    def extract_model_order(self, experiment_path: pathlib.Path) -> List[str]:
        """Extract the model order from the experiment directory name."""
        name = experiment_path.name
        
        # Parse experiment name to extract models
        # Format: prob_type_prob_id_beam_iters{N}_{simulator}_{model1}_{model2}_...
        parts = name.split('_')
        
        # Find where the models start (after simulator)
        models = []
        simulator_found = False
        
        for part in parts:
            if simulator_found:
                # Skip non-model parts (dropout, analyses, etc.)
                if part.startswith('dropout') or part.startswith('analyses') or \
                   part.startswith('plan') or part.startswith('code') or \
                   part.startswith('beam') or part.startswith('score') or \
                   part.startswith('util') or part.startswith('spadacc') or \
                   part.startswith('ancestors') or part.startswith('preventdupe') or \
                   part.startswith('planicl') or part.startswith('codeicl'):
                    break
                # This should be a model name
                models.append(part)
            elif part in ['firesim', 'spike', 'kernelbench']:
                simulator_found = True
        
        # Fallback: if we can't parse from directory name, use common default order
        if not models:
            logger.warning(f"Could not extract model order from {name}, using default order")
            models = ['o3-mini', 'gpt-4o']
        
        return models
    
    def extract_parent_performance_metrics(self, parent_candidate: Optional[CodeCandidate], experiment_info: Dict[str, str], hw_backend, prob: Prob) -> Dict[str, Any]:
        """Extract performance metrics from a parent CodeCandidate object.
        
        Since we assume parent code is always correct, we create performance metrics
        based on the available information in the CodeCandidate.
        """
        if not parent_candidate:
            return {"correct": False, "error": "Parent candidate not found"}
        
        # Create performance metrics assuming correctness
        performance_metrics = {
            "correct": True,
            "test_results": {}  # We don't have detailed test results, but assume all passed
        }
        
        # Add the score as the primary metric (latency for gemmini, runtime for cuda)
        if parent_candidate.score is not None and parent_candidate.score != float("inf"):
            if experiment_info['backend_type'] == 'gemmini':
                performance_metrics["latency"] = parent_candidate.score
            elif experiment_info['backend_type'] == 'cuda':
                performance_metrics["runtime"] = parent_candidate.score
        
        # Add spad_acc_stats feedback if available (string feedback messages)
        if hasattr(parent_candidate, 'spad_acc_stats') and parent_candidate.spad_acc_stats:
            performance_metrics["spad_acc_stats"] = parent_candidate.spad_acc_stats
        
        # Collect raw spad_acc utilization for Gemmini backend if parent code is available
        if (experiment_info['backend_type'] == 'gemmini' and 
            parent_candidate.code and 
            hasattr(hw_backend, 'get_spad_acc_utilization')):
            try:
                spad_acc_utilization_stats = hw_backend.get_spad_acc_utilization(prob, [parent_candidate.code])
                if spad_acc_utilization_stats:
                    util_stats = spad_acc_utilization_stats[0]
                    performance_metrics['spad_util'] = util_stats.get('spad_util', 0)
                    performance_metrics['acc_util'] = util_stats.get('acc_util', 0)
            except Exception as e:
                logger.warning(f"Error collecting parent spad_acc utilization: {e}")
        
        return performance_metrics
    
    def get_hardware_backend(self, backend_type: str, prob_type: str):
        """Get the appropriate hardware backend based on the backend type."""
        if backend_type == 'gemmini':
            # Determine PE dimensions based on problem type
            spad_size_kb = 256
            acc_size_kb = 64
            if "admm" in prob_type:
                pe_dim = 4
            elif "exo" in prob_type or "gemm" in prob_type:
                pe_dim = 16
            elif "gpt" in prob_type:
                pe_dim = 32
                spad_size_kb = 512
                acc_size_kb = 128
            else:
                pe_dim = 16  # Default
            return GemminiHardwareBackend(pe_dim, spad_size_kb, acc_size_kb)
        elif backend_type == 'cuda':
            return KBHardwareBackend()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def load_candidate_from_file(self, filepath: pathlib.Path) -> Optional[CodeCandidate]:
        """Load a CodeCandidate from a saved file."""
        try:
            content = filepath.read_text()
            
            # Handle backward compatibility - remove 'feedback' parameter if present
            # The 'feedback' parameter was removed from CodeCandidate.__init__()
            if 'feedback=' in content:
                import re
                # Remove feedback parameter and its value from the CodeCandidate constructor calls
                # Pattern matches: feedback=[...], or feedback=[...])
                content = re.sub(r',\s*feedback=\[[^\]]*\]', '', content)
                content = re.sub(r'feedback=\[[^\]]*\],?\s*', '', content)
            
            # The files contain Python code that creates CodeCandidate objects
            candidate = eval(content)
            return candidate
        except Exception as e:
            logger.error(f"Error loading candidate from {filepath}: {e}")
            return None
    
    def find_parent_candidate(self, experiment_path: pathlib.Path, iteration: int, candidate_idx: int) -> Tuple[Optional[str], Optional[CodeCandidate]]:
        """Find the parent candidate code and object for a given iteration and candidate index.
        
        Returns:
            Tuple of (parent_code, parent_candidate) where both can be None if not found
        """
        # Look for candidates from the previous iteration
        prev_iter = iteration - 1
        candidates_dir = experiment_path / f"candidates-iter-{prev_iter}"
        
        if not candidates_dir.exists():
            return None, None
        
        # First try exact filename match
        target_file = candidates_dir / f"candidate_{candidate_idx}.txt"
        if target_file.exists():
            candidate = self.load_candidate_from_file(target_file)
            return (candidate.code if candidate else None, candidate)
        
        # Fallback: use index-based lookup
        candidate_files = sorted(candidates_dir.glob("candidate_*.txt"))
        if candidate_idx < len(candidate_files):
            candidate = self.load_candidate_from_file(candidate_files[candidate_idx])
            return (candidate.code if candidate else None, candidate)
        
        logger.warning(f"Candidate index {candidate_idx} not found for iteration {iteration}")
        return None, None
    
    def get_global_plan_ordering(self, plan_dir: pathlib.Path, model_order: List[str]) -> List[pathlib.Path]:
        """Get all plan files ordered globally: first by candidate, then by model, then by plan index."""
        try:
            # Get all plan files
            all_plan_files = list(plan_dir.glob("plan_parent*_*.txt"))
            
            if not all_plan_files:
                return []
            
            # Sort plan files globally: candidate -> model -> plan_num
            def global_plan_sort_key(plan_file):
                parts = plan_file.stem.split('_')
                if len(parts) < 4:
                    return (999, 999, 999)  # Put malformed names at the end
                
                try:
                    candidate_idx = int(parts[1].replace('parent', ''))
                except ValueError:
                    candidate_idx = 999
                
                model = parts[2]  # Extract model name
                try:
                    plan_num = int(parts[3])
                except ValueError:
                    plan_num = 999
                
                # Model priority based on actual experiment order
                try:
                    model_priority = model_order.index(model)
                except ValueError:
                    # If model not found in order, put it at the end
                    model_priority = len(model_order)
                
                return (candidate_idx, model_priority, plan_num)
            
            all_plan_files.sort(key=global_plan_sort_key)
            return all_plan_files
            
        except Exception as e:
            logger.error(f"Error getting global plan ordering: {e}")
            return []

    def find_corresponding_plan(self, candidate_idx: int, plan_idx: int, plan_dir: pathlib.Path, model_order: List[str]) -> str:
        """Find the corresponding plan for given candidate and plan indices using global plan ordering.
        
        The plan_idx is a global index across all plans for this iteration, ordered as:
        1. First by candidate index (0, 1, 2, ...)
        2. Then by model order (as specified in model_order)
        3. Then by plan number generated by that model (0, 1, 2, ...)
        """
        try:
            # Get globally ordered plan files
            global_plan_files = self.get_global_plan_ordering(plan_dir, model_order)
            
            if not global_plan_files:
                return ""
            
            # Map global plan index to specific plan file
            if plan_idx < len(global_plan_files):
                selected_plan_file = global_plan_files[plan_idx]
                logger.debug(f"Global plan index {plan_idx} maps to file: {selected_plan_file.name}")
                return selected_plan_file.read_text().strip()
            else:
                logger.warning(f"Global plan index {plan_idx} out of range for {len(global_plan_files)} total plans")
                return ""
                
        except Exception as e:
            logger.error(f"Error finding plan for candidate {candidate_idx}, global plan index {plan_idx}: {e}")
            return ""
    
    def parse_implementation_file(self, impl_file: pathlib.Path, experiment_path: pathlib.Path, 
                                 iteration: int, plan_dir: pathlib.Path, model_order: List[str]) -> Optional[ImplementationInfo]:
        """Parse an implementation file and gather all related information."""
        try:
            # Extract implementation code
            impl_code = impl_file.read_text().strip()
            if not impl_code:
                return None
            
            # Parse filename: impl_a_b_{model}_{suffix}.txt
            impl_parts = impl_file.stem.split('_')
            if len(impl_parts) < 4:  # Need at least impl, a, b, model
                logger.warning(f"Invalid implementation filename format: {impl_file.name}")
                return None
            
            try:
                candidate_idx = int(impl_parts[1])  # 'a' - which candidate
                plan_idx = int(impl_parts[2])       # 'b' - global plan index across all plans for this iteration
                model = impl_parts[3]               # model name
            except ValueError:
                logger.warning(f"Could not extract indices from {impl_file.name}")
                return None
            
            # Find parent candidate code and object
            parent_code, parent_candidate = self.find_parent_candidate(experiment_path, iteration, candidate_idx)
            parent_code = parent_code or ""
            
            # Find corresponding plan
            plan_content = ""
            if plan_dir.exists():
                plan_content = self.find_corresponding_plan(candidate_idx, plan_idx, plan_dir, model_order)
            
            return ImplementationInfo(
                file_path=impl_file,
                code=impl_code,
                parent_code=parent_code,
                parent_candidate=parent_candidate,
                plan=plan_content,
                candidate_idx=candidate_idx,
                plan_idx=plan_idx,
                model=model
            )
            
        except Exception as e:
            logger.error(f"Error parsing {impl_file}: {e}")
            return None

    def find_generated_code_dirs(self) -> List[pathlib.Path]:
        """Find all generated-code-iter-x directories."""
        generated_code_dirs = []
        
        for output_dir in self.output_dirs:
            if not output_dir.exists():
                logger.warning(f"Output directory {output_dir} does not exist")
                continue
                
            # Find all experiment directories
            for experiment_dir in output_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue
                
                # Apply experiment filter if specified
                if self.experiment_filter and self.experiment_filter not in experiment_dir.name:
                    logger.debug(f"Skipping experiment (filter '{self.experiment_filter}'): {experiment_dir.name}")
                    continue
                
                logger.debug(f"Found experiment: {experiment_dir.name}")
                
                # Find all generated-code-iter-x directories in this experiment
                for subdir in experiment_dir.iterdir():
                    if subdir.is_dir() and subdir.name.startswith("generated-code-iter-"):
                        generated_code_dirs.append(subdir)
        
        return generated_code_dirs
    
    def process_generated_code_dir(self, code_dir: pathlib.Path) -> List[CodeIterationData]:
        """Process a single generated-code-iter-x directory using batch evaluation."""
        results = []
        experiment_path = code_dir.parent
        experiment_info = self.extract_experiment_info(experiment_path)
        
        # Extract iteration number from directory name
        match = re.search(r'generated-code-iter-(\d+)', code_dir.name)
        if not match:
            logger.error(f"Could not extract iteration number from {code_dir.name}")
            return results
        
        iteration = int(match.group(1))
        logger.info(f"Processing {experiment_info['experiment_name']} iteration {iteration}")
        
        # Extract model order from experiment configuration
        model_order = self.extract_model_order(experiment_path)
        logger.debug(f"Extracted model order: {model_order}")
        
        # Find corresponding plan directory
        plan_dir = experiment_path / f"generated-plans-iter-{iteration}"
        
        # Get hardware backend
        try:
            hw_backend = self.get_hardware_backend(experiment_info['backend_type'], experiment_info['prob_type'])
            prob = Prob(experiment_info['prob_type'], int(experiment_info['prob_id']))
        except Exception as e:
            logger.error(f"Error setting up backend for {experiment_path}: {e}")
            return results
        
        # Parse all implementation files (excluding _full.txt files)
        impl_files = [f for f in code_dir.glob("impl_*.txt") if not f.name.endswith("_full.txt")]
        impl_infos = []
        
        for impl_file in impl_files:
            impl_info = self.parse_implementation_file(impl_file, experiment_path, iteration, plan_dir, model_order)
            if impl_info:
                impl_infos.append(impl_info)
        
        if not impl_infos:
            logger.warning(f"No valid implementations found in {code_dir}")
            return results
        
        # Deduplicate implementations: only keep the first implementation for each (parent, plan, model) combination
        seen_combinations = set()
        deduplicated_impl_infos = []
        skipped_count = 0
        
        for impl_info in impl_infos:
            # Create a unique key from parent code, plan, and model
            combination_key = (impl_info.parent_code, impl_info.plan, impl_info.model)
            
            if combination_key not in seen_combinations:
                seen_combinations.add(combination_key)
                deduplicated_impl_infos.append(impl_info)
            else:
                skipped_count += 1
                logger.debug(f"Skipping duplicate implementation: {impl_info.file_path.name} (same parent, plan, model)")
        
        impl_infos = deduplicated_impl_infos
        logger.info(f"After deduplication: {len(impl_infos)} implementations remaining ({skipped_count} duplicates skipped)")
        
        if not impl_infos:
            logger.warning(f"No implementations remaining after deduplication in {code_dir}")
            return results
        
        # Process implementations in batches
        logger.info(f"Processing {len(impl_infos)} implementations in batches of {self.batch_size}")
        
        for batch_start in range(0, len(impl_infos), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(impl_infos))
            batch_infos = impl_infos[batch_start:batch_end]
            
            # Extract codes for batch evaluation
            batch_codes = [info.code for info in batch_infos]
            
            try:
                # Evaluate the batch of implementation codes
                batch_performance_metrics = hw_backend.evaluate_code(prob, batch_codes, experiment_info['simulator'])
                
                if not batch_performance_metrics:
                    batch_performance_metrics = [{"correct": False, "error": "No results returned"}] * len(batch_codes)
                elif len(batch_performance_metrics) != len(batch_codes):
                    logger.error(f"Batch evaluation returned {len(batch_performance_metrics)} results for {len(batch_codes)} codes")
                    batch_performance_metrics = [{"correct": False, "error": "Batch size mismatch"}] * len(batch_codes)
                
                # Collect spad_acc_stats for Gemmini backend (only for correct implementations)
                if experiment_info['backend_type'] == 'gemmini':
                    try:
                        # Only collect spad_acc_stats for correct implementations to avoid wasting time
                        correct_indices = [i for i, metrics in enumerate(batch_performance_metrics) if metrics.get('correct', False)]
                        if correct_indices:
                            correct_codes = [batch_codes[i] for i in correct_indices]
                            spad_acc_utilization_stats = hw_backend.get_spad_acc_utilization(prob, correct_codes)
                            
                            # Add spad_acc utilization to the corresponding performance metrics
                            for i, util_stats in enumerate(spad_acc_utilization_stats):
                                original_idx = correct_indices[i]
                                batch_performance_metrics[original_idx]['spad_util'] = util_stats.get('spad_util', 0)
                                batch_performance_metrics[original_idx]['acc_util'] = util_stats.get('acc_util', 0)
                    except Exception as spad_e:
                        logger.warning(f"Error collecting spad_acc_stats for batch: {spad_e}")
                
            except Exception as e:
                logger.error(f"Error evaluating batch starting at {batch_start}: {e}")
                batch_performance_metrics = [{"correct": False, "error": str(e)}] * len(batch_codes)
            
            # Create result data for each implementation in the batch
            for impl_info, performance_metrics in zip(batch_infos, batch_performance_metrics):
                # Extract parent performance metrics from the CodeCandidate object
                parent_performance_metrics = self.extract_parent_performance_metrics(
                    impl_info.parent_candidate, experiment_info, hw_backend, prob
                )
                
                result = CodeIterationData(
                    experiment_name=experiment_info['experiment_name'],
                    iteration=iteration,
                    parent_code=impl_info.parent_code,
                    plan=impl_info.plan,
                    implementation_code=impl_info.code,
                    performance_metrics=performance_metrics,
                    parent_performance_metrics=parent_performance_metrics,
                    file_path=str(impl_info.file_path),
                    prob_type=experiment_info['prob_type'],
                    prob_id=experiment_info['prob_id'],
                    simulator=experiment_info['simulator'],
                    backend_type=experiment_info['backend_type']
                )
                
                results.append(result)
                parent_score = parent_performance_metrics.get('latency') or parent_performance_metrics.get('runtime', 'N/A')
                impl_score = performance_metrics.get('latency') or performance_metrics.get('runtime', 'N/A')
                
                # Add spad_acc utilization info for Gemmini backend
                log_msg = f"Processed {impl_info.file_path.name}: impl_correct={performance_metrics.get('correct', False)}, parent_score={parent_score}, impl_score={impl_score}"
                if experiment_info['backend_type'] == 'gemmini':
                    parent_spad = parent_performance_metrics.get('spad_util', 'N/A')
                    parent_acc = parent_performance_metrics.get('acc_util', 'N/A')
                    impl_spad = performance_metrics.get('spad_util', 'N/A')
                    impl_acc = performance_metrics.get('acc_util', 'N/A')
                    log_msg += f", parent_spad={parent_spad:.3f}, parent_acc={parent_acc:.3f}, impl_spad={impl_spad:.3f}, impl_acc={impl_acc:.3f}" if isinstance(parent_spad, (int, float)) and isinstance(impl_spad, (int, float)) else f", spad_acc=N/A"
                
                logger.info(log_msg)
            
            logger.info(f"Completed batch {batch_start//self.batch_size + 1}/{(len(impl_infos) + self.batch_size - 1)//self.batch_size}")
        
        return results
    
    def save_intermediate_results(self, experiment_name: str, results: List[CodeIterationData]):
        """Save intermediate results for a specific experiment."""
        if not results:
            return
            
        # Create safe filename from experiment name
        safe_name = re.sub(r'[^\w\-_.]', '_', experiment_name)
        
        # Save as JSON
        json_file = self.intermediate_dir / f"{safe_name}.json"
        results_dict = [asdict(result) for result in results]
        
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save as pickle
        pickle_file = self.intermediate_dir / f"{safe_name}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved {len(results)} intermediate results for {experiment_name}")

    def save_results(self, output_file: str = "rerun_results.json"):
        """Save all results to a JSON file."""
        # Convert dataclass objects to dictionaries
        results_dict = [asdict(result) for result in self.results]
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved {len(results_dict)} results to {output_file}")
        
        # Also save as pickle for easier loading in Python
        pickle_file = output_file.replace('.json', '.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Saved results as pickle to {pickle_file}")
    
    def run_rerun_analysis(self):
        """Main method to run the complete rerun analysis."""
        logger.info("Starting code rerun analysis...")
        
        # Find all generated code directories
        code_dirs = self.find_generated_code_dirs()
        logger.info(f"Found {len(code_dirs)} generated-code directories")
        
        # Group directories by experiment for better intermediate saving
        experiment_dirs = {}
        for code_dir in code_dirs:
            experiment_path = code_dir.parent
            experiment_name = experiment_path.name
            if experiment_name not in experiment_dirs:
                experiment_dirs[experiment_name] = []
            experiment_dirs[experiment_name].append(code_dir)
        
        # Process each experiment
        for experiment_name, dirs in experiment_dirs.items():
            logger.info(f"Processing experiment: {experiment_name} ({len(dirs)} iterations)")
            experiment_results = []
            
            # Process each directory in this experiment
            for code_dir in dirs:
                try:
                    results = self.process_generated_code_dir(code_dir)
                    self.results.extend(results)
                    experiment_results.extend(results)
                except Exception as e:
                    logger.error(f"Error processing directory {code_dir}: {e}")
                    continue
            
            # Save intermediate results for this experiment
            if experiment_results:
                self.save_intermediate_results(experiment_name, experiment_results)
        
        logger.info(f"Completed analysis. Total results: {len(self.results)}")
        
        # Save final results
        self.save_results()
        
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics of the results."""
        if not self.results:
            logger.info("No results to summarize")
            return
        
        total_results = len(self.results)
        correct_results = sum(1 for r in self.results if r.performance_metrics.get('correct', False))
        parent_available = sum(1 for r in self.results if r.parent_performance_metrics.get('correct', False))
        
        # Count spad_acc_stats availability
        impl_spad_available = sum(1 for r in self.results if 'spad_util' in r.performance_metrics)
        parent_spad_available = sum(1 for r in self.results if 'spad_util' in r.parent_performance_metrics)
        
        logger.info(f"\n=== SUMMARY ===")
        logger.info(f"Total implementations processed: {total_results}")
        logger.info(f"Correct implementations: {correct_results}")
        logger.info(f"Success rate: {correct_results/total_results*100:.1f}%")
        logger.info(f"Parent performance data available: {parent_available}/{total_results} ({parent_available/total_results*100:.1f}%)")
        logger.info(f"Implementation spad_acc_stats available: {impl_spad_available}/{total_results} ({impl_spad_available/total_results*100:.1f}%)")
        logger.info(f"Parent spad_acc_stats available: {parent_spad_available}/{total_results} ({parent_spad_available/total_results*100:.1f}%)")
        
        # Group by experiment
        experiments = {}
        for result in self.results:
            exp_name = result.experiment_name
            if exp_name not in experiments:
                experiments[exp_name] = {'total': 0, 'correct': 0}
            experiments[exp_name]['total'] += 1
            if result.performance_metrics.get('correct', False):
                experiments[exp_name]['correct'] += 1
        
        logger.info(f"\nPer-experiment breakdown:")
        for exp_name, stats in experiments.items():
            success_rate = stats['correct']/stats['total']*100 if stats['total'] > 0 else 0
            logger.info(f"  {exp_name}: {stats['correct']}/{stats['total']} ({success_rate:.1f}%)")

def main():
    """Main function to run the rerun analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rerun all generated code iterations and save performance data")
    parser.add_argument("--output-dirs", nargs="+", default=["output", "output_old"],
                       help="Output directories to scan (default: output output_old)")
    parser.add_argument("--output-file", default="rerun_results.json",
                       help="Output file to save results (default: rerun_results.json)")
    parser.add_argument("--experiment-filter", default=None,
                       help="Filter string that must be present in experiment directory names (default: no filter)")
    
    args = parser.parse_args()
    
    # Create and run the rerunner
    rerunner = CodeRerunner(args.output_dirs, args.experiment_filter)
    rerunner.run_rerun_analysis()
    
    # Save to specified output file
    if args.output_file != "rerun_results.json":
        rerunner.save_results(args.output_file)

if __name__ == "__main__":
    main()