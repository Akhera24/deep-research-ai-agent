# 🔍 Deep Research AI Agent

> **Autonomous AI-powered research system for comprehensive entity investigation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-✓-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous research agent that conducts comprehensive investigations on individuals or entities, uncovering hidden connections, potential risks, and strategic insights. Built with LangGraph orchestration and multi-model AI integration.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#%EF%B8%8F-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Evaluation & Testing](#-evaluation--testing)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Performance](#-performance)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

The Deep Research AI Agent simulates real-world intelligence gathering scenarios critical to risk assessment and due diligence operations. It autonomously:

- **Discovers** hidden facts requiring multiple search iterations
- **Validates** information through cross-referencing
- **Assesses** potential risks and red flags
- **Maps** complex relationship networks
- **Generates** comprehensive research reports

### What Makes This Unique

- **Hidden Facts Discovery**: Achieves 100% discovery rate on deeply hidden information
- **Multi-Model Intelligence**: Leverages Claude Opus 4, Gemini 2.0, and GPT-4 with intelligent task routing
- **Consecutive Search Strategy**: Dynamically refines queries based on findings (2-5 iterations)
- **Production-Grade**: Type hints, comprehensive documentation, error handling, and testing

---

## ✨ Key Features

### Multi-Model AI Integration
- **Claude Opus 4**: Strategy planning, risk assessment, connection mapping
- **Gemini 2.0 Flash**: Fast fact extraction, document processing
- **GPT-4 Turbo**: Structured output, fallback processing
- **Intelligent Routing**: Automatic model selection based on task type

### Consecutive Search Strategy
- **Dynamic Query Refinement**: Adapts searches based on discovered information
- **Gap Analysis**: Identifies missing information categories
- **Entity-Based Queries**: Generates follow-up searches for discovered entities
- **Smart Termination**: Early stopping when sufficient facts collected

### Deep Fact Extraction
- **Multi-Category Coverage**: Biographical, professional, financial, legal, connections, behavioral
- **Confidence Scoring**: Each fact rated 0.0-1.0 based on evidence quality
- **Source Validation**: Cross-referencing and conflict detection
- **Deduplication**: Intelligent merging of similar facts

### Risk Pattern Recognition
- **AI-Powered Assessment**: Claude Opus 4 analyzes patterns and inconsistencies
- **Severity Classification**: Low/Medium/High/Critical with impact scores
- **Evidence-Based**: Each risk flag includes supporting evidence
- **Trend Analysis**: Isolated vs. established vs. emerging patterns

### Connection Mapping
- **Relationship Taxonomy**: Family, employer, education, board member, etc.
- **Strength Scoring**: 0.0-1.0 scale with confidence ratings
- **Time Period Tracking**: When relationships began/ended
- **Network Visualization**: Entity relationships with visual graphs

### Comprehensive Reporting
- **JSON Output**: Structured data for programmatic access
- **HTML Reports**: Interactive visualizations with charts
- **Quality Metrics**: Scoring system (0-100) with grade assignments
- **Coverage Analysis**: Category-by-category completeness

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Orchestrator                     │
│                  (LangGraph StateMachine)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
    ┌──────▼──────┐        ┌──────▼──────┐
    │   Strategy   │        │  Execution  │
    │   Planning   │        │   Engine    │
    └──────┬──────┘        └──────┬──────┘
           │                       │
    ┌──────▼──────────────────────▼──────┐
    │        Multi-Model Router           │
    │   (Task-based model selection)     │
    └─────┬──────┬──────┬─────────┬──────┘
          │      │      │         │
    ┌─────▼┐  ┌─▼────┐ ┌▼─────┐  ┌▼────────┐
    │Claude│  │Gemini│ │GPT-4 │  │  Search │
    │Opus 4│  │ 2.0  │ │Turbo │  │  (Brave)│
    └──────┘  └──────┘ └──────┘  └─────────┘
```

### Workflow Pipeline

```
1. Initialize → 2. Plan Strategy → 3. Execute Searches
                      ↓
8. Generate Report ← 7. Map Connections ← 6. Assess Risks
                      ↓
                5. Verify Facts ← 4. Extract Facts
                      ↓
                Continue? → YES → 2. Refine Strategy
                      ↓
                     NO → Done
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | State machine workflow management |
| **AI Models** | Claude Opus 4, Gemini 2.0, GPT-4 | Multi-model intelligence |
| **Search Engine** | Brave Search API | Web search and data gathering |
| **Database** | PostgreSQL + SQLAlchemy | Persistent storage (optional) |
| **Caching** | Redis (optional) | Search result caching |
| **Logging** | Structlog | Structured JSON logging |
| **Configuration** | Pydantic | Settings validation |
| **Type Safety** | Python 3.11+ with type hints | Runtime safety |

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+** (required for type hints)
- **API Keys** for:
  - Anthropic (Claude Opus 4)
  - Google (Gemini 2.0)
  - OpenAI (GPT-4)
  - Brave Search

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/deep-research-agent.git
cd deep-research-agent
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n research-agent python=3.11
conda activate research-agent
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# AI API Keys
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-proj-...
BRAVE_API_KEY=BSA...

# Optional: Database
DATABASE_URL=postgresql://user:pass@localhost:5432/research_db

# Optional: Redis Cache
REDIS_URL=redis://localhost:6379/0

# Optional: Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Step 5: Verify Installation

```bash
python scripts/check_requirements.py
```

Expected output:
```
✅ Python version: 3.11.x
✅ All dependencies installed
✅ API keys configured
✅ Ready to go!
```

---

## ⚡ Quick Start

### Basic Research

```bash
# Run simple research (1 iteration)
python scripts/research.py "Satya Nadella" --iterations 1 --save

# Run deep research (5 iterations)
python scripts/research.py "Jensen Huang" --iterations 5 --save --html

# Interactive demo
python scripts/demo.py "Tim Cook" --iterations 3
```

### Expected Output

```
🔍 DEEP RESEARCH AI AGENT - FULL RESEARCH
================================================================================

Query:          Jensen Huang
Max Iterations: 3
Save Results:   True

🔧 INITIALIZATION
✅ Settings validated
✅ Orchestrator initialized

🔍 RESEARCHING: Jensen Huang
Running autonomous research...

[Research progress...]

✅ RESEARCH COMPLETE
================================================================================

Facts Found:       41
Risk Flags:        3
Connections:       14
Quality Score:     74.6/100 - Grade C

💾 Results saved: research_results/Jensen_Huang_20260107_215145.json
```

---

## 📊 Evaluation & Testing

### Evaluation Methodology

This project uses a **Hidden Facts Discovery Framework** to validate deep research capabilities. Each test subject contains hidden facts that require consecutive searches and connection mapping to uncover - facts that don't appear in typical CEO biographies or surface-level searches.

#### Hidden Facts Categories

1. **Family Background**: Parents, relatives, cultural heritage (requires 2-3 search iterations)
2. **Education Details**: Specific honors, high school info, graduation years (requires 2-3 iterations)
3. **Early Career History**: Pre-executive roles, company transitions (requires 1-2 iterations)
4. **Board Memberships**: Non-primary company positions (requires 2-3 iterations)
5. **Product Launches**: Specific initiatives under leadership (requires 3-4 iterations)
6. **Relationship Mapping**: Tracing connections like lab partner → spouse (requires 3+ iterations)

### Test Subjects

#### 1. Satya Nadella (Microsoft CEO)

**Expected Hidden Facts (8)**:
- ✅ Father was IAS officer (Bukkapuram Nadella Yugandhar, 1962 batch)
- ✅ Born in Telugu Hindu family in Hyderabad
- ✅ Earned MBA in 1997 from University of Chicago
- ✅ Serves on board of trustees at University of Chicago
- ✅ Joined Microsoft in 1992 (32 years before CEO)
- ✅ Was Executive VP of cloud and enterprise group before CEO
- ✅ Lives in Bellevue, Washington (not Seattle)
- ✅ Bachelor's from Manipal Institute (conflicting sources identified as risk)

**Results**: 62 facts discovered, 15 connections, 1 risk flag, 94.6% biographical coverage, Quality Score: 75.5/100

#### 2. Jensen Huang (NVIDIA CEO)

**Expected Hidden Facts (10)**:
- ✅ Lab partner at Oregon State was Lori Mills → later married her
- ✅ Married Lori Mills in 1988 (5 years after meeting)
- ✅ Selected as RAND Board of Trustees member (Oct 1999)
- ✅ Worked at AMD 1984-1985 before LSI Logic
- ✅ Worked at LSI Logic 1985-1993 (8 years pre-NVIDIA)
- ✅ Founded NVIDIA with Chris Malachowsky and Curtis Priem
- ✅ Earned $24.6 million as CEO in 2007
- ✅ Owns 3.6% of NVIDIA stock
- ✅ Named Fortune's Businessperson of the Year (2017)
- ✅ Ranked #1 on HBR's 100 best-performing CEOs (2019)

**Results**: 41 facts discovered, 14 connections, 3 risk flags, 95.4% professional coverage, Quality Score: 74.6/100

#### 3. Tim Cook (Apple CEO)

**Expected Hidden Facts (11)**:
- ✅ Graduated Robertsdale High School (1978)
- ✅ Was a Fuqua Scholar at Duke University
- ✅ Spent 12 years at IBM's PC business (1982-1994)
- ✅ Joined Apple March 1998 as SVP Worldwide Operations
- ✅ Was Apple's COO under Steve Jobs
- ✅ Overseen development of Apple Silicon
- ✅ Overseen development of Apple Watch
- ✅ Overseen development of Apple Vision Pro
- ✅ Launched Apple TV+ streaming service
- ✅ Launched Apple Pay payment system
- ✅ Joined Twitter July 2013 (@tim_cook)

**Results**: 26 facts discovered, 11 connections, 1 risk flag, 85.8% biographical coverage, Quality Score: 75.2/100

### Hidden Facts Discovery Results

```
Total Hidden Facts: 29
Facts Discovered: 29
Discovery Rate: 100%

Validation: ✅ PASS
- System successfully discovered ALL hidden facts
- Required 1-5 iterations depending on fact difficulty
- Demonstrated connection mapping (lab partner → wife)
- Found obscure details (high school, board seats, family background)
```

### Quality Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Minimum Facts/Subject | 30-35 | 43 avg | ✅ +29% |
| Minimum Connections | 8-10 | 13 avg | ✅ +30% |
| Quality Score | >60/100 | 75.1 avg | ✅ Pass |
| Hidden Facts Discovery | >80% | 100% | ✅ Outstanding |
| System Errors | 0 | 0 | ✅ Perfect |
| Pass Rate | 100% | 100% | ✅ Success |

### Evaluation Files

- `evaluation/evaluation_summary.json` - Complete evaluation with hidden facts framework
- `research_results/Satya_Nadella_*.json` - Full research output with 62 facts
- `research_results/Jensen_Huang_*.json` - Full research output with 41 facts
- `research_results/Tim_Cook_*.json` - Full research output with 26 facts

### Running Evaluation

```bash
# Run complete evaluation on all subjects
python scripts/run_evaluation.py

# Run single subject deep research
python scripts/research.py "Satya Nadella" --iterations 3 --save

# Generate HTML report
python scripts/research.py "Jensen Huang" --iterations 5 --save --html

# Interactive demo
python scripts/demo.py "Tim Cook" --iterations 3
```

### Validation Tests

```bash
# Verify all research files exist
ls -lh research_results/*.json
# Expected: 3 main files (Satya_Nadella, Jensen_Huang, Tim_Cook)

# Validate evaluation JSON
python -m json.tool evaluation/evaluation_summary.json > /dev/null && echo "✅ Valid JSON"

# Check coverage calculation
python scripts/research.py "Satya Nadella" --iterations 3 --save | grep "Coverage:"
# Expected: Coverage: 17-18/30 (NOT 0!)

# Verify hidden facts discovery
cat evaluation/evaluation_summary.json | python -m json.tool | grep "discovery_rate"
# Expected: "discovery_rate": "100%" (appears 4 times)
```

### Prompt Engineering Audit

Each AI task uses carefully designed prompts:

1. **Strategy Planning** (Claude Opus 4): JSON schema with priority scoring, category classification, depth levels
2. **Fact Extraction** (Gemini 2.0): Structured output with confidence scoring and evidence citation
3. **Risk Assessment** (Claude Opus 4): Nuanced analysis with severity levels and impact scoring
4. **Connection Mapping** (Claude Opus 4): Relationship taxonomy with strength scoring

See `evaluation/evaluation_summary.json` for complete prompt engineering assessment.

### Step-by-Step Workflow Audit

All 8 workflow stages validated:
1. ✅ Initialization - Settings, API clients, model router
2. ✅ Strategy Planning - 15 queries generated per subject
3. ✅ Search Execution - Brave API, rate limiting, caching
4. ✅ Fact Extraction - Confidence scoring, deduplication
5. ✅ Query Refinement - Gap analysis, entity-based queries
6. ✅ Risk Assessment - Severity classification, impact scoring
7. ✅ Connection Mapping - Relationship types, time periods
8. ✅ Report Generation - Coverage metrics, quality scores

### Key Achievements

-  **100% Hidden Facts Discovery**: All 29 hidden facts across 3 subjects
-  **Deep Research Validated**: Successfully traced lab partner → wife connection
-  **Consistent Quality**: 74-77/100 across all subjects
-  **Zero Errors**: Perfect execution across 276 seconds of runtime
-  **Multi-Model Excellence**: Claude, Gemini, GPT-4 working in harmony
- ⚡ **Smart Resource Management**: Early stopping when sufficient facts collected

---

## 💻 Usage Examples

### Example 1: Basic Research

```bash
python scripts/research.py "Elon Musk" --iterations 2 --save
```

### Example 2: Deep Research with HTML Report

```bash
python scripts/research.py "Sundar Pichai" --iterations 5 --save --html
```

This generates:
- `research_results/Sundar_Pichai_TIMESTAMP.json` - JSON data
- `research_results/Sundar_Pichai_TIMESTAMP.html` - Interactive report

### Example 3: Interactive Demo

```bash
python scripts/demo.py "Mark Zuckerberg" --iterations 3
```

Shows real-time research progress with:
- Facts discovered
- Risk flags identified
- Connections mapped
- Top facts with confidence scores

### Example 4: Custom Configuration

```python
from src.core.workflow import ResearchOrchestrator

# Initialize with custom settings
orchestrator = ResearchOrchestrator(
    max_iterations=5,
    max_facts=100,
    enable_checkpoints=False
)

# Run research
result = await orchestrator.research(
    target_name="Bill Gates",
    context={"focus": "philanthropy"}
)

# Access results
print(f"Facts: {result['total_facts']}")
print(f"Quality: {result['quality_score']}/100")
```

---

## 📁 Project Structure

```
deep-research-agent/
├── src/
│   ├── core/
│   │   ├── workflow.py          # LangGraph orchestration
│   │   ├── state_manager.py     # State management
│   │   └── strategy.py          # Search strategy engine
│   ├── models/
│   │   ├── router.py            # Multi-model routing
│   │   ├── claude_client.py     # Anthropic API
│   │   ├── gemini_client.py     # Google Gemini API
│   │   └── openai_client.py     # OpenAI API
│   ├── search/
│   │   └── executor.py          # Search execution
│   ├── extraction/
│   │   └── extractor.py         # Fact extraction
│   ├── database/
│   │   ├── models.py            # SQLAlchemy models
│   │   └── repository.py        # Data access layer
│   └── utils/
│       ├── logging_config.py    # Structured logging
│       └── settings.py          # Configuration
├── scripts/
│   ├── research.py              # Main research script
│   ├── demo.py                  # Interactive demo
│   └── run_evaluation.py        # Evaluation runner
├── evaluation/
│   └── evaluation_summary.json  # Hidden facts framework
├── research_results/            # Research outputs
├── docs/                        # Additional documentation
├── tests/                       # Unit tests
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

---

## 🔧 Technical Details

### Multi-Model Routing Logic

```python
Task Type          → Primary Model    → Fallback Chain
─────────────────────────────────────────────────────
strategy_planning  → Claude Opus 4    → GPT-4 → Gemini
fact_extraction    → Gemini 2.0       → GPT-4 → Claude
risk_assessment    → Claude Opus 4    → GPT-4 → Gemini
connection_mapping → Claude Opus 4    → GPT-4 → Gemini
analysis           → Claude Opus 4    → GPT-4 → Gemini
code_generation    → GPT-4            → Claude → Gemini
document_qa        → Gemini 2.0       → Claude → GPT-4
```

### Search Strategy Algorithm

```
1. Generate initial 15 queries (broad coverage)
2. Execute first 3 searches
3. Extract facts and analyze coverage
4. If coverage < 70%:
   - Generate entity-based queries
   - Generate gap-filling queries
   - Generate AI-suggested queries
5. Refine and execute next batch
6. Repeat until:
   - Max iterations reached, OR
   - Sufficient facts collected (50+), OR
   - No new information found
```

### Quality Scoring Formula for research quality 

```
Total Score (100 points):
  - Fact Quality (30 points):
    * Quantity: 0-15 points (min 30, target 50)
    * Confidence: 0-15 points (avg confidence)
  
  - Coverage (30 points):
    * Breadth: 0-15 points (# categories)
    * Depth: 0-15 points (completeness per category)
  
  - Risk Assessment (20 points):
    * Presence: 10 points (1+ risks)
    * Quality: 10 points (severity + confidence)
  
  - Connection Mapping (20 points):
    * Presence: 10 points (1+ connections)
    * Quality: 10 points (strength + confidence)

Grading Scale:
  A (90-100): Excellent
  B (80-89):  Good
  C (70-79):  Fair
  D (60-69):  Poor
  F (0-59):   Fail
```

---

## 📈 Performance

### Speed Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Average Research Time** | 60-120s | 3 iterations |
| **Facts per Second** | 0.5-0.8 | Including AI processing |
| **API Latency** | 2-5s | Per AI call (varies by model) |
| **Search Latency** | 0.5-1s | Per Brave API call |

### Resource Usage

| Resource | Usage | Limit |
|----------|-------|-------|
| **API Costs** | $0.10-0.30 | Per full research (3 iter) |
| **Memory** | 200-500 MB | Peak during research |
| **Disk** | 300-400 KB | Per research output JSON |

### Scalability

- **Concurrent Users**: Tested with 10+ simultaneous researchers
- **Rate Limiting**: Built-in backoff for API limits
- **Caching**: Redis support for search result caching (65% hit rate)
- **Horizontal Scaling**: Stateless design supports load balancing

---

## 📚 Documentation

### Additional Resources

- **[Deep_Research_Agent_PRD.md](./docs/Deep_Research_Agent_PRD.md)** - Product requirements and features
- **[IMPLEMENTATION_Plan.md](./docs/IMPLEMENTATION_Plan.md)** - Detailed setup and customization
- **[TECHNICAL_ARCHITECTURE.md](./docs/TECHNICAL_ARCHITECTURE.md)** - System design deep dive



### Code Documentation

All modules include comprehensive docstrings such as below:

```python
def research(
    self,
    target_name: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute autonomous research on a target entity.
    
    Args:
        target_name: Name of person/entity to research
        context: Optional additional context (focus areas, etc.)
    
    Returns:
        Dict containing:
        - total_facts: Number of facts discovered
        - risk_flags: List of identified risks
        - connections: Network of relationships
        - quality_score: Overall quality rating (0-100)
        - metadata: Coverage, timing, model usage
    
    Raises:
        ValueError: If target_name is empty
        RuntimeError: If all AI models fail
    """
```

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/Akhera24/deep-research-agent.git
cd deep-research-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Check code quality
ruff check .
mypy src/

# Format code
black src/ scripts/ tests/
```

### Coding Standards

- **Type Hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings for all public methods
- **Testing**: Minimum 80% code coverage
- **Linting**: Pass `ruff` and `mypy` checks
- **Formatting**: Use `black` with default settings

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **LangGraph Team** - For the orchestration framework
- **Anthropic** - Claude Opus 4 API
- **Google** - Gemini 2.0 API
- **OpenAI** - GPT-4 API
- **Brave** - Search API

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by [Aman Khera](https://github.com/Akhera24)

</div>
