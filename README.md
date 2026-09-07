# DEIMOS

> **An environment-aware autonomous computer agent built around closed-loop execution, dynamic environment memory, independent verification, and bounded recovery.**

**Observe → Plan → Act → Verify → Recover**

---

## Overview

**DEIMOS** is an experimental autonomous computer agent designed to interact with a computer through a **state-aware, closed-loop control architecture**.

Rather than treating task execution as a simple sequence of:

```text
User Request → LLM → Tool Call → "Task Completed"
```

DEIMOS treats execution as a feedback process.

The agent continuously reasons about the relationship between:

* the user's requested objective,
* the current environment state,
* the action being executed,
* the resulting state transition,
* and whether the intended outcome was actually achieved.

The core execution loop is therefore:

```text
                 ┌──────────────────┐
                 │    USER GOAL     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ UNDERSTAND GOAL  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ OBSERVE STATE    │◄───────────────┐
                 └────────┬─────────┘                │
                          │                          │
                          ▼                          │
                 ┌──────────────────┐                │
                 │ PLAN / SELECT    │                │
                 │ CAPABILITY       │                │
                 └────────┬─────────┘                │
                          │                          │
                          ▼                          │
                 ┌──────────────────┐                │
                 │ POLICY CHECK     │                │
                 └────────┬─────────┘                │
                          │                          │
                          ▼                          │
                 ┌──────────────────┐                │
                 │ EXECUTE ACTION   │                │
                 └────────┬─────────┘                │
                          │                          │
                          ▼                          │
                 ┌──────────────────┐                │
                 │ OBSERVE RESULT   │────────────────┘
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ VERIFY OUTCOME   │
                 └────────┬─────────┘
                          │
                   ┌──────┴──────┐
                   │             │
                 PASS          FAILURE
                   │             │
                   ▼             ▼
                COMPLETE     RECOVER / REPLAN
```

The objective is not simply to make an AI capable of triggering actions.

The objective is to build an agent capable of **observing the consequences of its actions and using those observations to determine what to do next**.

---

# Why DEIMOS?

Most basic AI automation systems have a structural weakness:

> They assume that an action being executed means the intended task was successfully completed.

DEIMOS separates these concepts.

For DEIMOS:

```text
Action Attempted ≠ Action Completed ≠ Objective Achieved
```

An agent can:

* call the correct function,
* execute a command without errors,
* receive a successful process exit code,

and still fail to achieve the user's actual objective.

DEIMOS therefore introduces a control architecture built around four fundamental principles:

### 1. Environment awareness

The agent should reason about the actual environment rather than blindly executing assumptions.

### 2. Closed-loop execution

Actions should be followed by observation of their consequences.

### 3. Independent verification

The system should distinguish between an action completing and the intended outcome being achieved.

### 4. Bounded recovery

Failures should trigger controlled recovery attempts rather than unlimited retries or hallucinated success.

---

# Core Capabilities

## 💬 Conversational Interaction

DEIMOS accepts natural-language requests through a conversational interface.

The agent can distinguish between requests that require:

* conversational reasoning,
* environment inspection,
* project operations,
* file operations,
* supported computer actions,
* or task execution.

The conversational layer acts as an interface between natural language and the structured execution runtime.

---

## 🖥️ Environment Awareness

DEIMOS operates with awareness of relevant environment state.

Instead of executing actions purely from a static plan, the agent can inspect the state surrounding supported workflows and use that information during execution.

The basic control principle is:

```text
Observe
   ↓
Act
   ↓
Observe Again
   ↓
Evaluate State Transition
```

This makes the agent fundamentally different from a static automation pipeline.

The environment is treated as a dynamic system whose state can change between actions.

---

## 🔄 Closed-Loop Execution

DEIMOS is built around a feedback-driven execution loop.

A simplified execution path is:

```text
Goal
 ↓
Observe
 ↓
Plan
 ↓
Policy
 ↓
Execute
 ↓
Observe
 ↓
Verify
 ↓
Recover if Required
```

This creates a continuous relationship between the agent's reasoning and the actual environment.

The agent does not simply issue a sequence of actions and assume success.

Instead, execution produces new information that can influence subsequent decisions.

---

## 📂 Dynamic Environment Memory

DEIMOS includes persistent memory for environment information, particularly file and project locations.

The memory system can be used to:

* retain known project locations,
* locate previously discovered files,
* resolve named projects,
* search indexed environment information,
* refresh location information as the filesystem changes.

This allows the agent to operate without requiring the user to repeatedly provide absolute paths.

For example:

```text
User:
"Open my ML project."

        ↓

DEIMOS:
Search dynamic environment memory

        ↓

Resolve matching project location

        ↓

Verify availability

        ↓

Execute supported open operation
```

The memory layer is intended to represent **persistent environment knowledge**, rather than simply extending the conversation context window.

---

## 📁 Project and File Operations

DEIMOS currently supports structured project-related workflows including:

* locating projects,
* resolving named projects,
* opening projects,
* opening files,
* creating directories,
* creating project structures,
* preparing supported development environments.

These capabilities are executed through the agent's structured control pipeline rather than being exposed only as isolated utility functions.

---

## 💻 Development Environment Setup

DEIMOS can perform supported development setup workflows.

Current capabilities include workflows related to:

* project initialization,
* directory creation,
* project structure setup,
* environment preparation,
* opening projects in supported development tooling.

The system currently includes support for workflows involving **Visual Studio Code**.

The focus is on goal-oriented development operations rather than simple shell command execution.

---

# Autonomous Execution Model

DEIMOS is currently best described as a:

> **Capability-bounded autonomous computer agent.**

This distinction is intentional.

DEIMOS already performs autonomous execution inside the boundaries of its implemented capabilities.

However, it is not currently presented as an unrestricted agent capable of performing arbitrary computer tasks across every application and environment.

Its autonomy exists through the following process:

```text
Natural Language Goal
        ↓
Goal Interpretation
        ↓
Environment Observation
        ↓
Capability / Task Selection
        ↓
Planning
        ↓
Policy Evaluation
        ↓
Execution
        ↓
State Observation
        ↓
Verification
        ↓
Recovery or Completion
```

This makes DEIMOS more than a static command router while avoiding unsupported claims of unrestricted computer autonomy.

---

# Policy-Controlled Execution

Planning and execution are deliberately separated.

A model or planner can propose an action, but the proposed action does not automatically receive execution authority.

The control path follows:

```text
Planner Output
      ↓
Proposed Action
      ↓
Policy Validation
      ↓
Allowed / Rejected
      ↓
Execution
```

This architecture creates a separation between:

* what the planner wants to do,
* what the system permits,
* and what is actually executed.

This separation becomes increasingly important as agent capabilities expand.

---

# Independent Outcome Verification

DEIMOS does not rely exclusively on the language model's own confidence to determine whether a task succeeded.

The execution pipeline distinguishes between:

```text
Action Attempted
        ↓
Action Executed
        ↓
Result Observed
        ↓
Outcome Verified
```

A successful function call is therefore not automatically equivalent to a successful task.

Verification provides a mechanism for detecting conditions such as:

* the expected resource was not created,
* the expected application did not open,
* the environment did not transition into the required state,
* execution completed but produced an incorrect result.

The final result can therefore be based on observable evidence rather than the agent simply declaring:

> "Done."

---

# Bounded Recovery

Failures are not immediately treated as terminal.

When execution or verification indicates failure, DEIMOS can enter a controlled recovery process.

```text
Execution
    ↓
Failure Detected
    ↓
Recovery Attempt
    ↓
Re-Observe Environment
    ↓
Re-Verify Outcome
```

Recovery is intentionally bounded.

The agent should not enter uncontrolled loops where it repeatedly performs the same failing action without making progress.

The general objective is:

```text
Failure
   ↓
Diagnose
   ↓
Attempt Recovery
   ↓
Observe
   ↓
Verify
   ↓
Complete or Abort
```

---

# 🧠 Planner Architecture

DEIMOS separates planning from the underlying model provider.

The project currently includes a planner abstraction with support for:

* mock planning,
* OpenAI-compatible planning backends.

This makes it possible to test the runtime independently from live LLM infrastructure.

The planner is responsible for structured reasoning about supported actions, while execution remains controlled by the runtime.

Conceptually:

```text
                 ┌─────────────────┐
                 │      Planner    │
                 └────────┬────────┘
                          │
                    Structured Plan
                          │
                          ▼
                 ┌─────────────────┐
                 │ Runtime / Policy│
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Execution    │
                 └─────────────────┘
```

This separation allows planning behavior and execution behavior to evolve independently.

---

# 🎙️ Voice Interaction

DEIMOS includes components for voice-based interaction.

The speech subsystem contains support for:

* microphone input,
* speech-to-text,
* text-to-speech.

The voice pipeline is modular and can be tested independently from the primary computer-control runtime.

Conceptually:

```text
Voice Input
    ↓
Speech-to-Text
    ↓
Conversation / Agent Runtime
    ↓
Response
    ↓
Text-to-Speech
```

---

# 👁️ Experimental Vision Support

DEIMOS includes experimental vision-related infrastructure and a vision-oriented baseline.

The broader research direction explores the relationship between:

* structured machine-readable environment state,
* direct system interaction,
* and vision-based computer interaction.

The project does not treat vision as the only source of environment information.

Instead, the architecture investigates a broader control model where the agent can prefer reliable structured state when available and use visual information where appropriate.

---

# Benchmarking and Evaluation

The repository includes benchmarking infrastructure and a vision-oriented baseline.

The purpose of this infrastructure is to evaluate execution strategies rather than relying only on demonstrations.

The evaluation framework is intended to support comparison between approaches such as:

```text
Vision-Oriented Execution
          VS
Structured / State-Aware Execution
```

Relevant evaluation concepts include:

* verified task success,
* failure detection,
* false-success conditions,
* recovery attempts,
* execution behavior,
* and task-level outcomes.

This is important because a computer agent can appear impressive in a demonstration while still failing silently under repeated evaluation.

---

# Architecture

DEIMOS is organized into several major components.

```text
                         ┌──────────────────────┐
                         │        USER          │
                         │    Text / Voice      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   CONVERSATION API   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       PLANNER        │
                         │ Goal Interpretation  │
                         └──────────┬───────────┘
                                    │
                                    ▼
              ┌───────────────────────────────────────┐
              │         ENVIRONMENT OBSERVATION        │
              │                                       │
              │ File / Project / Window / System State│
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                         ┌──────────────────────┐
                         │       POLICY         │
                         │ Action Authorization │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       RUNNER         │
                         │ Execution Controller │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      OS / TASKS      │
                         │ Computer Interaction │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    OBSERVATION       │
                         │ State After Action   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         ▼                      ▼
                 ┌──────────────┐       ┌──────────────┐
                 │ VERIFICATION │       │   RECOVERY   │
                 └──────┬───────┘       └──────┬───────┘
                        │                      │
                        └──────────┬───────────┘
                                   ▼
                         ┌──────────────────────┐
                         │   FINAL RESPONSE     │
                         └──────────────────────┘
```

---

# Project Structure

```text
DEIMOS/
│
├── agent_control/
│   │
│   ├── planner/
│   │   ├── base.py
│   │   ├── mock.py
│   │   └── openai_compat.py
│   │
│   ├── platform_window/
│   │   ├── _linux.py
│   │   └── _win32.py
│   │
│   ├── speech/
│   │   ├── mic.py
│   │   ├── stt.py
│   │   └── tts.py
│   │
│   ├── tasks/
│   │   ├── open_project.py
│   │   └── setup_project.py
│   │
│   ├── api.py
│   ├── conversation.py
│   ├── memory.py
│   ├── observe.py
│   ├── os_tools.py
│   ├── policy.py
│   ├── recovery.py
│   ├── response.py
│   ├── runner.py
│   ├── session.py
│   ├── task.py
│   ├── trace.py
│   ├── types.py
│   ├── verifiers.py
│   └── vision_fallback.py
│
├── benchmark/
│   ├── baselines/
│   │   └── vision_only.py
│   │
│   ├── tasks/
│   │   ├── group_a.py
│   │   └── open_named.py
│   │
│   ├── harness.py
│   └── inject.py
│
├── projects/
│
├── scripts/
│   ├── smoke_setup_project.py
│   └── smoke_voice_open_project.py
│
├── tests/
│
├── .env.example
├── main.py
├── pytest.ini
└── requirements.txt
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/SakshamJuneja007/deimos.git
cd deimos
```

> Replace the repository URL if the GitHub repository has not yet been renamed.

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configuration

Create a local environment configuration from the example file.

### Windows

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Configure the required model provider settings.

The planner is designed around OpenAI-compatible APIs, allowing different compatible model providers to be used.

> **Never commit `.env` or API keys to the repository.**

---

# Usage

## Start DEIMOS

```bash
python main.py
```

---

## Start a conversational session

```bash
python main.py chat
```

Disable speech output:

```bash
python main.py chat --no-speak
```

---

## Run system diagnostics

```bash
python main.py doctor
```

For an offline diagnostic:

```bash
python main.py doctor --offline
```

---

## Dynamic memory

Refresh the environment memory index:

```bash
python main.py memory --refresh
```

Search stored environment information:

```bash
python main.py memory "project name"
```

---

## List supported tasks

```bash
python main.py tasks
```

---

## Run tests

```bash
pytest -q
```

---

# Current Capability Status

## Implemented

* [x] Conversational interaction
* [x] Environment awareness for supported workflows
* [x] Closed-loop execution
* [x] Dynamic environment memory
* [x] Persistent project/file location knowledge
* [x] Project discovery and resolution
* [x] Opening supported projects and files
* [x] Folder and project setup operations
* [x] Development environment workflows
* [x] Visual Studio Code integration for supported workflows
* [x] Policy-controlled execution
* [x] Independent outcome verification
* [x] Bounded recovery
* [x] Execution tracing
* [x] Speech components
* [x] Pluggable planner architecture
* [x] Benchmark infrastructure
* [x] Experimental vision baseline

---

## Current Boundary

DEIMOS is **not currently presented as an unrestricted general-purpose computer agent**.

Its autonomous behavior operates within the set of capabilities implemented by the runtime.

The next major research and engineering challenge is expanding from:

```text
Known Goal
   ↓
Known Capability
   ↓
Structured Execution
```

toward:

```text
Unfamiliar Goal
      ↓
Environment Analysis
      ↓
Dynamic Capability Selection
      ↓
General Planning
      ↓
Closed-Loop Execution
      ↓
Verification
      ↓
Recovery / Replanning
```

while preserving the existing reliability architecture.

---

# Design Principles

## State Before Assumption

The agent should prefer observed environment state over assumptions about what the computer should look like.

---

## Execution Is Not Success

Completing an action does not prove that the user's objective was achieved.

---

## Observe After Acting

Actions change the environment.

The agent should inspect those changes.

---

## Verify Independently

The agent's own confidence is not sufficient evidence of success.

---

## Recover, Don't Hallucinate

Failures should produce controlled recovery behavior or explicit failure reporting—not fabricated success.

---

## Keep Autonomy Bounded

Increasing agent autonomy without maintaining control and verification eventually produces unreliable behavior.

DEIMOS prioritizes controlled expansion of autonomy rather than unrestricted action generation.

---

# Roadmap

Future development directions include:

* [ ] General goal decomposition beyond current workflows
* [ ] Dynamic action selection
* [ ] Broader tool abstractions
* [ ] More general computer interaction
* [ ] Expanded environment observation
* [ ] Stronger replanning mechanisms
* [ ] Additional recovery strategies
* [ ] Broader application support
* [ ] Expanded benchmark suites
* [ ] Structured-state and vision hybrid routing
* [ ] More advanced persistent memory
* [ ] Improved cross-platform execution

---

# Project Status

**Experimental / Active Development**

DEIMOS is currently focused on building and evaluating reliable closed-loop mechanisms for computer-agent execution.

The project prioritizes:

```text
Reliability
    over
Feature Count

Verification
    over
Model Confidence

Observed State
    over
Assumed State

Measured Behavior
    over
Impressive Demos
```

---

# DEIMOS

> **Observe the environment. Execute deliberately. Verify the outcome. Recover when necessary.**

```text
OBSERVE
   ↓
PLAN
   ↓
ACT
   ↓
VERIFY
   ↓
RECOVER
```

---

## Author

**Saksham Juneja**

GitHub: `@SakshamJuneja007`
