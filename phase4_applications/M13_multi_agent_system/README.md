# M13: Multi-Agent System

## Problem Statement

Design and build a multi-agent system where multiple LLM-backed agents collaborate to solve a task. Define agent roles, a communication protocol, and orchestration logic. Demonstrate that the system accomplishes tasks that benefit from specialization and coordination.

## Mathematical Formulation

### Role Definition
(Placeholder for agent types, capabilities, and constraints)

### Communication Protocol
(Placeholder for message format, routing, and turn-taking)

### Orchestration
(Placeholder for coordinator logic, handoff conditions, aggregation)

## Intuition

(Placeholder for intuition on when multi-agent helps vs single-agent, and how to reduce coordination overhead)

## Implementation Plan

1. Choose a task that benefits from multiple roles (e.g., research: searcher, analyzer, writer)
2. Define 2-4 agent roles with distinct responsibilities
3. Design message format: sender, receiver, content, metadata
4. Implement coordinator: assigns tasks, routes messages, aggregates outputs
5. Implement per-agent logic: receive context, generate response, send to coordinator or another agent
6. Define termination condition (e.g., coordinator decides task is done)
7. Test on representative tasks
8. Document protocol and example interaction trace

(No code.)

## Required Experiments

- Run system on 3-5 multi-step tasks
- Document agent interactions (who said what, in what order)
- Compare multi-agent vs single-agent on the same tasks (quality, cost, latency)
- (Optional) Ablate: number of agents, coordination strategy

## Required Visualizations

- System architecture diagram: agents, coordinator, communication flow
- Example interaction trace (sequence diagram or transcript)
- Table: task, multi-agent result, single-agent result, comparison

## Reflection Questions

1. When does adding more agents improve outcomes vs add noise?
2. How do you prevent agents from contradicting each other or repeating work?
3. What are the tradeoffs between centralized coordination and peer-to-peer communication?

## Completion Checklist

- [ ] Multi-agent system implemented with defined roles and protocol
- [ ] Example tasks and traces documented in `experiments_log.md`
- [ ] Architecture diagram in `visualizations/`
- [ ] Journal entry in `learning_journal.md`
