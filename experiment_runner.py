import asyncio
from typing import Dict, Any, Optional, List
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from datetime import datetime
import json
import logging
from pathlib import Path
import docker
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentResult:
    """Contains experiment execution results"""
    experiment_id: str
    status: str
    start_time: datetime
    end_time: datetime
    metrics: Dict[str, Any]
    outputs: List[Dict[str, Any]]
    error: Optional[str] = None

class ExperimentRunner:
    def __init__(self, workspace_dir: str = "experiments"):
        """Initialize experiment runner"""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        self.docker_client = docker.from_env()
        self.active_experiments: Dict[str, ExperimentResult] = {}

    def _create_notebook(self, code: str, env_vars: Optional[Dict[str, str]] = None) -> nbformat.NotebookNode:
        """Create a Jupyter notebook from code"""
        notebook = nbformat.v4.new_notebook()
        
        # Add environment setup if needed
        if env_vars:
            setup_code = [
                "import os",
                "# Set environment variables",
                *[f"os.environ['{k}'] = '{v}'" for k, v in env_vars.items()]
            ]
            notebook.cells.append(nbformat.v4.new_code_cell(source="\n".join(setup_code)))

        # Add main code cell
        notebook.cells.append(nbformat.v4.new_code_cell(source=code))
        return notebook

    async def run_experiment(
        self,
        code: str,
        experiment_id: str,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: int = 600
    ) -> ExperimentResult:
        """Run a code experiment in an isolated environment"""
        try:
            # Initialize result
            result = ExperimentResult(
                experiment_id=experiment_id,
                status="running",
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics={},
                outputs=[],
            )
            self.active_experiments[experiment_id] = result

            # Create experiment directory
            exp_dir = self.workspace_dir / experiment_id
            exp_dir.mkdir(exist_ok=True)

            # Create and save notebook
            notebook = self._create_notebook(code, env_vars)
            notebook_path = exp_dir / "experiment.ipynb"
            with open(notebook_path, 'w') as f:
                nbformat.write(notebook, f)

            # Run in container
            container = self.docker_client.containers.run(
                "jupyter/scipy-notebook:latest",
                command=[
                    "jupyter", "nbconvert",
                    "--execute",
                    "--to", "notebook",
                    "--ExecutePreprocessor.timeout=-1",
                    "/workspace/experiment.ipynb"
                ],
                volumes={
                    str(exp_dir.absolute()): {
                        'bind': '/workspace',
                        'mode': 'rw'
                    }
                },
                working_dir='/workspace',
                mem_limit='4g',
                cpu_count=2,
                detach=True
            )

            # Wait for completion
            container.wait(timeout=timeout)

            # Process results
            if container.attrs['State']['ExitCode'] == 0:
                # Load executed notebook
                with open(notebook_path, 'r') as f:
                    executed_nb = nbformat.read(f, as_version=4)

                # Collect outputs
                outputs = []
                for cell in executed_nb.cells:
                    if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                        for output in cell.outputs:
                            if output.output_type in ['stream', 'display_data', 'execute_result']:
                                outputs.append({
                                    'type': output.output_type,
                                    'content': output.get('text', output.get('data', {}))
                                })

                # Update result
                result.status = "completed"
                result.end_time = datetime.now()
                result.outputs = outputs
                result.metrics = {
                    'execution_time': (result.end_time - result.start_time).total_seconds(),
                    'memory_used': container.attrs['HostConfig']['Memory'],
                    'exit_code': container.attrs['State']['ExitCode']
                }
            else:
                result.status = "failed"
                result.error = container.logs().decode('utf-8')

            # Cleanup
            container.remove()
            return result

        except Exception as e:
            logger.error(f"Error running experiment {experiment_id}: {str(e)}")
            result.status = "failed"
            result.error = str(e)
            result.end_time = datetime.now()
            return result

    def get_experiment_status(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get status of an experiment"""
        return self.active_experiments.get(experiment_id)

    def visualize_results(self, result: ExperimentResult, output_path: Path):
        """Create visualizations of experiment results"""
        if not result.metrics:
            return

        # Create basic metrics plot
        plt.figure(figsize=(10, 6))
        metrics_df = pd.DataFrame([result.metrics])
        metrics_df.plot(kind='bar')
        plt.title(f"Experiment {result.experiment_id} Metrics")
        plt.tight_layout()
        plt.savefig(output_path / "metrics.png")
        plt.close()

        # Save execution summary
        summary = {
            'experiment_id': result.experiment_id,
            'status': result.status,
            'execution_time': str(result.end_time - result.start_time),
            'metrics': result.metrics,
            'error': result.error
        }
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    # Example usage
    async def main():
        runner = ExperimentRunner()
        code = """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Generate some data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        # Create a plot
        plt.plot(x, y)
        plt.title('Simple Sine Wave')
        plt.show()
        
        # Print some metrics
        print(f"Data points: {len(x)}")
        print(f"Max value: {y.max():.2f}")
        """
        
        result = await runner.run_experiment(
            code=code,
            experiment_id="test_experiment",
            env_vars={'DISPLAY': 'test'}
        )
        
        print(f"Experiment status: {result.status}")
        print(f"Execution time: {result.metrics.get('execution_time')}s")
        if result.error:
            print(f"Error: {result.error}")

    asyncio.run(main())
