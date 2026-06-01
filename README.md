# Multi-Agent Orchestrator

A lightweight framework for coordinating multiple AI agents with shared state, sequential execution, and parallel fan-out.

## Why This Exists

Single-agent systems hit limits when tasks need multiple perspectives. A researcher, critic, and writer working together produce better output than one generalist agent. This framework makes multi-agent coordination explicit and debuggable.

## Features

- **Shared State**: All agents read/write to a shared context object
- **Sequential Pipelines**: Agent A output feeds into Agent B input
- **Parallel Fan-Out**: Run multiple agents simultaneously, merge results
- **Retry & Fallback**: Per-agent retry with configurable fallback strategies
- **Observability**: Full trace of every agent's input/output for debugging

## Quick Start

```python
from orchestrator import Orchestrator, Agent

# Define agents
researcher = Agent(name="researcher", task="find 3 facts about {topic}")
writer = Agent(name="writer", task="summarize these facts: {researcher_output}")

# Run pipeline
orch = Orchestrator(agents=[researcher, writer])
result = orch.run_sequential(topic="climate change")
print(result)
```

## Architecture

Built around a simple abstraction: agents are functions with a `name`, `task` template, and optional `dependencies`. The orchestrator resolves the dependency graph and executes accordingly.
