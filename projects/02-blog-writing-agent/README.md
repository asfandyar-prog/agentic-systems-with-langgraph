# ✍️ Project 02 — Blog Writing Agent (LangGraph)

## Overview

This project implements a structured blog-writing agent using LangGraph.

The system follows a graph-based architecture to:
- Generate structured blog sections
- Maintain state across iterations
- Compose long-form content modularly

## Architecture

Graph Flow:

START → Planner → Section Generator → Aggregator → END

### State Fields

- `topic`
- `sections`
- `current_section`
- `final_blog`

## Key Concepts Demonstrated

- Typed State with LangGraph
- Node-based content generation
- Controlled state mutation
- Multi-step content assembly

## Example Input

```python
{
    "topic": "Write a blog on Self Attention",
    "sections": []
}