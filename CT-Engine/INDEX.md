# Documentation Index

Welcome to the Semantic Parser Framework documentation! This index will help you find what you need.

## 📚 Quick Navigation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** ⭐ START HERE
  - 5-minute setup guide
  - First examples
  - Common patterns
  - Troubleshooting

### Complete Overview
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** 
  - What the framework is
  - Complete feature list
  - File structure
  - Design principles

### User Guide
- **[README.md](README.md)**
  - Comprehensive documentation
  - Installation instructions
  - Configuration guide
  - API reference
  - Best practices

### Development Guides
- **[BUILDING_ACTIONS.md](BUILDING_ACTIONS.md)**
  - Action lifecycle
  - Design patterns
  - Common use cases
  - Testing strategies
  - Advanced techniques

### Technical Details
- **[ARCHITECTURE.md](ARCHITECTURE.md)**
  - System architecture
  - Component details
  - Data flow
  - Design patterns
  - Extensibility points

## 🎯 Find What You Need

### "I want to get started quickly"
→ Read **QUICKSTART.md** (5 minutes)
→ Run `python example_usage.py`

### "I need to understand what this framework does"
→ Read **PROJECT_SUMMARY.md** (10 minutes)

### "I want to create my first custom action"
→ Read **BUILDING_ACTIONS.md** (20 minutes)
→ Use **custom_actions_template.py** as starting point

### "I need complete documentation"
→ Read **README.md** (30 minutes)

### "I want to understand the design"
→ Read **ARCHITECTURE.md** (30 minutes)

### "I need working code examples"
→ Check **example_usage.py**
→ Look at **example_actions.py**
→ Use **custom_actions_template.py**

## 📁 File Overview

### Core Package (`semantic_parser/`)
```
action_protocol.py      - Action interface and registry
llm_client.py          - Azure OpenAI client
reasoning_step.py      - Data models for reasoning
prompt_builder.py      - Prompt construction
react_engine.py        - Main orchestration engine
example_actions.py     - Example implementations
```

### Documentation
```
QUICKSTART.md          - Quick start guide
PROJECT_SUMMARY.md     - Complete overview
README.md             - Full documentation
BUILDING_ACTIONS.md   - Action development
ARCHITECTURE.md       - Technical details
```

### Examples and Templates
```
example_usage.py              - Working examples
custom_actions_template.py    - Action templates
```

### Configuration
```
.env.example          - Environment template
requirements.txt      - Dependencies
setup.py             - Package installation
```

## 🎓 Learning Path

### Beginner Path
1. **QUICKSTART.md** - Get it running (5 min)
2. **example_usage.py** - See it work (5 min)
3. **PROJECT_SUMMARY.md** - Understand the basics (10 min)
4. **custom_actions_template.py** - Create first action (20 min)

### Intermediate Path
1. Complete Beginner Path
2. **README.md** - Full features (30 min)
3. **BUILDING_ACTIONS.md** - Advanced actions (30 min)
4. **example_actions.py** - Study examples (15 min)

### Advanced Path
1. Complete Intermediate Path
2. **ARCHITECTURE.md** - Design deep dive (30 min)
3. Source code - Read implementation (60 min)
4. Extend framework - Build custom features

## 🔍 Topic Index

### Installation & Setup
- QUICKSTART.md → Installation
- README.md → Installation
- .env.example → Configuration

### Basic Usage
- QUICKSTART.md → Basic Usage
- README.md → Quick Start
- example_usage.py → All examples

### Creating Actions
- BUILDING_ACTIONS.md → Complete guide
- custom_actions_template.py → Templates
- example_actions.py → Examples
- README.md → Creating Custom Actions

### Architecture & Design
- ARCHITECTURE.md → Complete architecture
- PROJECT_SUMMARY.md → Design principles
- README.md → Core Components

### Advanced Topics
- BUILDING_ACTIONS.md → Advanced Techniques
- ARCHITECTURE.md → Extensibility Points
- README.md → Advanced Features

### API Reference
- README.md → Core Components
- Source code → Inline documentation
- __init__.py → Public API exports

### Troubleshooting
- QUICKSTART.md → Troubleshooting
- README.md → Troubleshooting
- BUILDING_ACTIONS.md → Testing & Debugging

## 💡 Common Questions

### How do I get started?
**Answer**: Read QUICKSTART.md, copy .env.example to .env, add your credentials, run example_usage.py

### How do I create a custom action?
**Answer**: See BUILDING_ACTIONS.md section "Action Design Patterns", use custom_actions_template.py

### What actions are available?
**Answer**: See example_actions.py for examples, create your own following the templates

### How does the reasoning work?
**Answer**: See ARCHITECTURE.md "Data Flow" section and README.md "How it Works"

### Can I use a different LLM?
**Answer**: Yes! See ARCHITECTURE.md "Custom LLM Clients" section

### How do I debug issues?
**Answer**: Set verbose=True in ReACTEngine, see BUILDING_ACTIONS.md "Debugging Actions"

### What are the design patterns used?
**Answer**: See ARCHITECTURE.md "Design Patterns Used" section

### How do I extend the framework?
**Answer**: See ARCHITECTURE.md "Extensibility Points" and BUILDING_ACTIONS.md

## 📖 Reading Order by Goal

### Goal: Get It Working
1. QUICKSTART.md
2. example_usage.py

### Goal: Build Something
1. QUICKSTART.md
2. BUILDING_ACTIONS.md
3. custom_actions_template.py

### Goal: Understand Everything
1. PROJECT_SUMMARY.md
2. README.md
3. ARCHITECTURE.md
4. BUILDING_ACTIONS.md

### Goal: Contribute
1. Complete "Understand Everything" path
2. Read source code
3. See ARCHITECTURE.md "Future Enhancements"

## 🔗 External Resources

### ReACT Paper
- Original paper describing the ReACT pattern
- Search: "ReACT: Synergizing Reasoning and Acting in Language Models"

### Azure OpenAI
- Documentation: https://learn.microsoft.com/azure/ai-services/openai/
- API Reference: https://platform.openai.com/docs/api-reference

### Pydantic
- Documentation: https://docs.pydantic.dev/

## 📝 Documentation Standards

All documentation in this framework follows these standards:

### Structure
- Clear headings and sections
- Table of contents for long documents
- Progressive disclosure (simple → complex)

### Code Examples
- Syntax highlighted
- Complete and runnable
- Commented where necessary

### Explanations
- Clear and concise
- Practical examples
- Common pitfalls noted

## 🔄 Documentation Updates

This framework is version 0.1.0. Documentation is current as of this version.

When extending the framework:
1. Update relevant documentation files
2. Add examples to example_usage.py
3. Update this index if needed

## ✅ Quick Reference

| Need | File | Time |
|------|------|------|
| Setup | QUICKSTART.md | 5 min |
| Overview | PROJECT_SUMMARY.md | 10 min |
| Full docs | README.md | 30 min |
| Actions | BUILDING_ACTIONS.md | 30 min |
| Design | ARCHITECTURE.md | 30 min |
| Examples | example_usage.py | 10 min |
| Templates | custom_actions_template.py | - |

## 🎯 Success Checklist

- [ ] Read QUICKSTART.md
- [ ] Set up .env file
- [ ] Run example_usage.py successfully
- [ ] Understand the output
- [ ] Read PROJECT_SUMMARY.md
- [ ] Create first custom action
- [ ] Parse your first query
- [ ] Read README.md for full features

## 📬 Questions?

If you can't find what you need:
1. Check this index again
2. Search within the documentation files
3. Read the source code (it's well-commented!)
4. Review the examples

---

**Happy parsing!** Start with [QUICKSTART.md](QUICKSTART.md) to get running in 5 minutes! 🚀
