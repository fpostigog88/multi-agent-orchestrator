"""
Multi-Agent Orchestrator
Coordinate multiple AI agents with shared state, sequential pipelines, and parallel execution.
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


@dataclass
class AgentResult:
    """Result from a single agent execution."""
    agent_name: str
    status: str  # "success", "error", "timeout"
    output: Any
    duration_ms: int
    error_message: Optional[str] = None


@dataclass
class OrchestratorRun:
    """Complete trace of a multi-agent run."""
    run_id: str
    status: str
    results: Dict[str, AgentResult] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    total_duration_ms: int = 0


class Agent:
    """
    A single agent in the orchestration graph.
    
    Agents are defined by:
    - name: unique identifier
    - task: template string that can reference shared_state or other agents' outputs
    - dependencies: list of agent names that must complete before this one runs
    - max_retries: how many times to retry on failure
    """
    
    def __init__(
        self,
        name: str,
        task: str,
        dependencies: List[str] = None,
        max_retries: int = 2,
        timeout_ms: int = 30000
    ):
        self.name = name
        self.task = task
        self.dependencies = dependencies or []
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms
    
    def execute(self, shared_state: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's task against the current shared state.
        In production, this would call an LLM API.
        """
        start = time.time()
        
        try:
            # Format task template with shared state
            task_input = self.task
            for key, value in shared_state.items():
                placeholder = f"{{{key}}}"
                if placeholder in task_input:
                    task_input = task_input.replace(placeholder, str(value))
            
            # Simulate agent work (replace with actual LLM call)
            time.sleep(random.uniform(0.1, 0.5))  # Simulate latency
            
            # Generate mock output based on task
            output = f"[{self.name}] Processed: {task_input[:100]}..."
            
            duration = int((time.time() - start) * 1000)
            
            return AgentResult(
                agent_name=self.name,
                status="success",
                output=output,
                duration_ms=duration
            )
            
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return AgentResult(
                agent_name=self.name,
                status="error",
                output=None,
                duration_ms=duration,
                error_message=str(e)
            )


class Orchestrator:
    """
    Orchestrates multiple agents with dependency resolution.
    
    Two execution modes:
    - Sequential: agents run in order, each sees previous outputs
    - Parallel: independent agents run simultaneously, then merge
    """
    
    def __init__(self, agents: List[Agent]):
        self.agents = {a.name: a for a in agents}
        self.validate_graph()
    
    def validate_graph(self):
        """Check for circular dependencies."""
        visited = set()
        path = set()
        
        def visit(name):
            if name in path:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            path.add(name)
            visited.add(name)
            
            agent = self.agents.get(name)
            if agent:
                for dep in agent.dependencies:
                    if dep not in self.agents:
                        raise ValueError(f"Agent {name} depends on unknown agent {dep}")
                    visit(dep)
            
            path.remove(name)
        
        for name in self.agents:
            visit(name)
    
    def run_sequential(self, **inputs) -> OrchestratorRun:
        """
        Run agents in dependency order. Each agent's output is added to shared state.
        """
        start = time.time()
        run_id = f"run_{int(start * 1000)}"
        
        state = dict(inputs)
        results = {}
        
        # Topological sort by dependencies
        executed = set()
        pending = set(self.agents.keys())
        
        while pending:
            ready = [
                name for name in pending
                if all(dep in executed for dep in self.agents[name].dependencies)
            ]
            
            if not ready:
                raise ValueError("Dependency resolution failed - possible cycle")
            
            for name in ready:
                agent = self.agents[name]
                
                # Retry loop
                for attempt in range(agent.max_retries + 1):
                    result = agent.execute(state)
                    
                    if result.status == "success":
                        break
                    
                    if attempt < agent.max_retries:
                        time.sleep(0.5 * (attempt + 1))
                
                results[name] = result
                
                if result.status == "success":
                    state[f"{name}_output"] = result.output
                
                executed.add(name)
                pending.remove(name)
        
        duration = int((time.time() - start) * 1000)
        
        return OrchestratorRun(
            run_id=run_id,
            status="success",
            results=results,
            shared_state=state,
            total_duration_ms=duration
        )
    
    def run_parallel(self, **inputs) -> OrchestratorRun:
        """
        Run all agents in parallel threads. No inter-agent dependencies in output.
        """
        start = time.time()
        run_id = f"run_{int(start * 1000)}"
        
        state = dict(inputs)
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(agent.execute, state): name
                for name, agent in self.agents.items()
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=30)
                    results[name] = result
                    if result.status == "success":
                        state[f"{name}_output"] = result.output
                except Exception as e:
                    results[name] = AgentResult(
                        agent_name=name,
                        status="error",
                        output=None,
                        duration_ms=0,
                        error_message=str(e)
                    )
        
        duration = int((time.time() - start) * 1000)
        
        return OrchestratorRun(
            run_id=run_id,
            status="success",
            results=results,
            shared_state=state,
            total_duration_ms=duration
        )


if __name__ == "__main__":
    # Demo: research pipeline
    researcher = Agent(
        name="researcher",
        task="Research facts about {topic}",
        max_retries=2
    )
    
    writer = Agent(
        name="writer",
        task="Write summary based on {researcher_output}",
        dependencies=["researcher"],
        max_retries=1
    )
    
    critic = Agent(
        name="critic",
        task="Review {writer_output} for accuracy",
        dependencies=["writer"],
        max_retries=1
    )
    
    orch = Orchestrator(agents=[researcher, writer, critic])
    
    print("=== Sequential Run ===")
    result = orch.run_sequential(topic="artificial intelligence")
    
    for name, agent_result in result.results.items():
        print(f"\n{name}: {agent_result.status} ({agent_result.duration_ms}ms)")
        print(f"  Output: {agent_result.output}")
    
    print(f"\nTotal duration: {result.total_duration_ms}ms")
