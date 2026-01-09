---
name: hugging-face-toad
description: Agent Builder to build customised powerful agents to help Users with their tasks. Use to build and configure agents in this environment - for tasks including search, analysis and general tasks. Can be combined with the hugging-face-tool-builder to create powerful HF Specific integrations.
---

# Agent Builder

The purpose of this skill is to create one or more AgentCards that will be installed in this environment to help the User achieve their goals.

AgentCards are markdown files with a simple YAML frontmatter that describe Agents. Good agents:
 - Goal focussed - will direct themselves using tools and available information to reach a goal.
 - Simple - but use all available tools and techniques to produce high quality outputs. 
 - Flexible - Able to adapt to unexpected circumstances to achieve the outcome.

This skill is describes how to:
 - Customising Agents for Users to interact with
 - Creating context-efficient sub-agents (as Tools) for Agents to use when completing higher level tasks.

AgentCards support preloading of User/Assistant conversation in to the Agents context for in-context learning/few-shot prompting, or providing a starting point for future User/Agent conversation turns.

Examples include but are not limited to:
 - Creating "experts" that have direct access to information from URLs or specific areas of the filesystem.
 - Building Tools to automate tasks and workflows.
 - Using in-context learning (few shot prompting) to rewrite documents, prose, source code or otherwise transform content. 

## Loading and Using AgentCards

### On Startup

AgentCards placed in the `.fast-agent/agent-cards` directory are automatically loaded on startup, and are accessible to the User. This is the simplest approach to install a new AgentCard.

AgentCards placed in the `.fast-agent/tool-cards` directory are loaded as Tools for the default agent.

**CRITICAL** In the `Toad` client environment, the application must be restarted before new or modified Agents are visible to the User. The User can use `ctrl+o` to "change mode" and see the available agents.

AgentCards can be hot-loaded as Tools for the current Agent by the User with the `/card <filename.md> --tool` slash command. 

## AgentCard Format

### Frontmatter

A typical frontmatter looks like this:

```yaml
---
name: agent_name # name for the agent - if not supplied the filename is used (preferred).
description: # optional, description used if this AgentCard is used as a tool
agents: # optional, list of agents  
---
```

Here is a short reference of the main configuration options. Don't specify a value if the default is preferred.

#### Core Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | filename | AgentCard identifier. Defaults to the filename (without extension) if omitted (preferred). |
| `description` | string | - | Description used when the AgentCard is exposed as a tool. |
| `default` | boolean | false | Marks this agent as the default when multiple agents are loaded. |
| `model` | string | config default | Model selector - prefer default unless User specifies preference |


#### Behavior

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_history` | boolean | true | Retain conversation history between turns. Set to `false` for stateless "responder" agents. |
| `shell` | boolean | false | Enable shell command execution via `execute` tool. |
| `cwd` | string | - | Working directory for shell commands (requires `shell: true`). |
| `skills` | list/string | auto | Skills directory or `[]` to disable skill discovery. (See note below) |


#### Agents as Tools

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agents` | list | [] | Child agents available as tools to this agent. |

#### History Preload

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | string/list | - | Path(s) to history file(s) for preloading conversation turns. Use the `./messages/` subdirectory to avoid confusion with AgentCards. **NOTE** Argument is *relative* to the agent-cards directory |

### Advanced

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `function_tools` | list | [] | Python functions as tools: `["module.py:function_name"]`. (See note below) |
| `request_params` | dict | {} | Model request parameters (e.g., `{temperature: 0.7}`). Use defaults (DO NOT SPECIFY) unless the User requests specific settings.  |


### Body

The optional markdown body following the frontmatter defines the agent's system prompt. If not specifed, the default is used.

## System Prompt

The System Prompt supports the following placeholders:

| Placeholder | Description | Behavior on Error |
|-------------|-------------|-------------------|
| `{{currentDate}}` | Current date (e.g., "17 December 2025") | N/A - always succeeds |
| `{{hostPlatform}}` | Platform info (e.g., "Linux-6.6.0-x86_64") | N/A - always succeeds |
| `{{pythonVer}}` | Python version (e.g., "3.12.0") | N/A - always succeeds |
| `{{workspaceRoot}}` | Working directory path | Empty string if not set |
| `{{env}}` | Environment description | Empty string if not set |
| `{{serverInstructions}}` | MCP server instructions (XML) | Empty string if no servers/failure |
| `{{agentSkills}}` | Available skills (XML) | Empty string if no skills |
| `{{file:path}}` | Read file content (relative path) | Raises error if missing |
| `{{file_silent:path}}` | Read file content (relative path) | Empty string if missing. This is useful for optional files like AGENTS.md |
| `{{url:https://...}}` | Fetch content from URL | Empty string on failure |

Notes

- File paths in {{file:...}} must be relative (absolute paths raise an error)
- {{file_silent:AGENTS.md}} is commonly used to include project-specific agent documentation
- Unresolved placeholders are replaced with empty string (won't confuse the LLM)

### Default System Prompt

If no System Prompt is supplied, the default is used: 

```markdown
You are a helpful AI Agent.

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}

The current date is {{currentDate}}.
```

### Location

System Prompts can be supplied either by:

- Placing the System Prompt content directly in the AgentCard.
- Loaded from a file or URL using `instruction: <path|url>` 

**TIP** When including content from URLs, access files in their raw form or as markdown whenever possible e.g.:
 - GitHub: `https://raw.githubusercontent.com/kepano/obsidian-skills/refs/heads/main/json-canvas/SKILL.md`
 - Gist: `https://gist.githubusercontent.com/evalstate/e5d659bbde778c526531d1cd8bc3d933/raw/d9e52d0de9dcada4d7a0a8605944e52428345d3a/test.py`
 - Hugging Face: `https://huggingface.co/spaces/toad-hf-inference-explorers/README/raw/main/README.md`
 - General Documentation: `https://modelcontextprotocol.io/docs/learn/architecture.md `

**TIP** Including content from URLs can simplify managing behaviour and rules across multiple deployed Agents (e.g. by placing rules or information in Gists or repositories).

## Preloading Message History

Message History can be loaded from a markdown file:

- If the first non‑empty line is not a delimiter, the whole file is treated as a single user message (simple mode).
- Otherwise, messages are delimited by exact, case‑sensitive markers on their own line:
    - ---USER
    - ---ASSISTANT
    - ---RESOURCE

- ---RESOURCE attaches a file to the current user/assistant section; the next line must be a  resource path (relative to the history file). The loader reads the file and embeds it.

Example:

```
---USER
Summarize the attached config.

---RESOURCE
configs/app.yaml

---ASSISTANT
Here’s a brief summary...
```

How to use it in an AgentCard:

```
---
name: my_agent
messages: ./message/history.md
---
(Optional system prompt here)
```

## Agents as Tools

Agents can be supplied as Tools to a parent Agent. 

For example, a "Search Agent" would be useful as a Tool as the search and summarization operations would be managed within the child agents context. The "Search Agent" may also use a faster, more efficient model and itself have tools or shell access to support it's function.

To include an Agent as a Tool, place it in the "Agents" list using it's name:

```
agents:
  - tool1
  - tool2
```

Use this pattern where an Agent may conduct tasks that may take multiple turns (for example searching) and only the result is required in the parent agent context.

## Design Considerations

### Model Selection

By default, Agents will use the default model supplied by the harness, which is capable of the required tasks. There may be instances where a specific model is required or requested. That can be selected by including `model: <model_id>` in the frontmatter.

For tasks that require speed `gpt-oss` is a good, relatively cheap choice, especially for search and "workhorse" style tool calling/data processing.

### Shell Access

Agents can be given access to the shell with `shell: true` in the frontmatter. This adds an `execute` tool that allows the Agent to execute bash commands. Note in the Toad environment shell access will be present.

### Python Functions

An alternative to direct shell access is to equip the Agent with Python functions. 

You reference functions using the format module.py:function_name in the function_tools field:

---
name: my_agent
function_tools:
  - tools.py:add
  - hf_api_tool.py:hf_api_request
---

The path is resolved relative to the AgentCard's directory, so you can keep the function file next to your card.

#### How It Works

1. When the card loads, the loader parses the module.py:function_name spec
2. Uses importlib.util to dynamically load the Python module
3. Extracts the named function and wraps it as a FastMCP tool
4. The tool's JSON schema is auto-generated from type hints and docstring

| Aspect | Constraint |
|--------|------------|
| Type hints | Required on parameters for schema generation |
| Return type | Should be JSON-serializable |
| Execution | Runs in the same Python process (no sandboxing) |
| Module syntax | Must be valid Python (errors fail fast) |
| Callable check | Must be a callable (raises AgentConfigError if not) |

#### How Dependencies Work

Dependencies work via standard Python imports - your function module can import anything installed in the environment:

```python 
import json
import os
from urllib.request import Request, urlopen

def hf_api_request(endpoint: str, method: str = "GET") -> dict:
    token = os.getenv("HF_TOKEN")  # Environment variables work
    # ... use urllib, json, etc.
```

Key points:
- All imports in your module are executed when loaded
- External packages must be installed in the same environment as fast-agent
- Environment variables are accessible via os.getenv()
- Import failures raise AgentConfigError with clear messages

Python functions can be faster, more robust and efficient at processing and transforming data.

### History Management

By default Agents retain their history, making them conversational. 

Agents as Tools do not retain history between tool invocations. 

To disable history retention for an agent, use `history: false` in the frontmatter. This is extremely useful for Agents that should act as "Responders". Particular use-cases for this are:
- Agents that should answer questions without being skewed by previous conversation turns.
- Agents configured with User/Assistant turns for in-context learning/few-shot prompting where the goal of the Agent is  content transformation to a particular pattern - for example would be adjusting prose or writing style.

### Interaction with Agent Skills

Agent Skills are markdown files named SKILL.md containing instructions and examples and so on that the Agent may read if to extend its capabilities.

Disable skills for Agents that have a very specific focussed task with a narrow outcome.

Specify a skills directory to constrain available skills. For example in the AgentCard frontmatter: `skills: ./my-custom-skills/` and include a SKILLS.md file.  

Skills descriptions are automatically loaded in to the System Prompt if {{agentSkills}} is present.

#### Disabling Skills

By default, if Agent Skills are present in the .fast-agent/skills directory, they are automatically loaded in to the System Prompt and shell access is enabled. If you DO NOT want the Agent to automatically use and discover skills, disable them in the AgentCard frontmatter: `skills: []`. 

Skills may also be disabled by excluding {{agentSkills}} from the System Prompt, however this will generate a warning if skills were otherwise available.


