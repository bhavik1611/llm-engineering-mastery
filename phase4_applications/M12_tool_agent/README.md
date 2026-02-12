# M12: Tool Agent

## Problem Statement

Build an agent that uses an LLM to decide which tools to call and with what arguments. The agent receives user requests, generates tool calls (e.g., via structured output or parsing), executes tools, and incorporates tool results into the next turn until the task is complete or a final answer is produced.

## Mathematical Formulation

### Tool Representation
(Placeholder for tool schema, parameters, return type)

### Agent Loop
(Placeholder for state transition: prompt + history + tool results to next action)

### Parsing and Execution
(Placeholder for extraction of tool name and arguments from model output)

## Intuition

(Placeholder for intuition on when to call tools vs respond directly, and how tool results affect subsequent reasoning)

## Implementation Plan

1. Define 2-4 tools with clear interfaces (e.g., search, calculator, database lookup)
2. Create tool descriptions for the prompt (name, description, parameters)
3. Implement agent loop: prompt with tools, generate, parse output
4. Parse model output for tool name and arguments (JSON or similar)
5. Execute tool and append result to conversation
6. Repeat until model returns final answer or max iterations
7. Handle parsing failures and tool errors gracefully
8. Test on tasks that require tool use (e.g., "What is the population of X?")

(No code.)

## Required Experiments

- Run agent on 5-10 tasks that require tool use
- Document success rate and failure modes (parse errors, wrong tool, wrong args)
- (Optional) Compare structured output (JSON) vs free-form parsing
- (Optional) Test with tools that return large or noisy outputs

## Required Visualizations

- Agent loop diagram: user input, model, parse, execute, iterate
- Table: task, tools called, success/failure, notes

## Reflection Questions

1. How do you design tool descriptions to improve tool selection accuracy?
2. What happens when the model hallucinates tool names or arguments?
3. When should the agent stop calling tools and return an answer?

## Completion Checklist

- [ ] Agent loop implemented with at least 2 tools
- [ ] Tasks documented in `experiments_log.md` with success/failure
- [ ] Failure modes and mitigations documented
- [ ] Journal entry in `learning_journal.md`
