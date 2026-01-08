"""
Deep Research AI Agent - Interactive Demo

Features:
- ASCII visualizations
- Interactive fact viewing
- HTML report generation
- Robust error handling
- Support for both dict and object-style facts
- Complete documentation

Usage examples:
    python scripts/demo.py "Tim Cook"                      # Quick summary
    python scripts/demo.py "Tim Cook" --full               # Show everything
    python scripts/demo.py "Tim Cook" --iterations 5       # More thorough
    python scripts/demo.py "Tim Cook" --save              # Save JSON
    python scripts/demo.py "Tim Cook" --html              # Generate HTML report
    python scripts/demo.py "Tim Cook" --full --html --save # Complete output

"""

import asyncio
import sys
import os
import json
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.workflow import ResearchOrchestrator
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ==============================================================================
# ASCII ART & VISUALIZATIONS
# ==============================================================================

BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              🔍 DEEP RESEARCH AI AGENT - INTERACTIVE DEMO 🔍              ║
║                                                                           ║
║        Multi-Model AI | Risk Assessment | Connection Mapping            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


def print_section_header(title: str, symbol: str = "═"):
    """Print a beautiful section header"""
    width = 80
    print(f"\n{symbol * width}")
    print(f"{title.center(width)}")
    print(f"{symbol * width}\n")


def safe_get(item: Union[Dict, Any], key: str, default: Any = None) -> Any:
    """
    Safely get attribute from either dict or object.
    
    Handles both dictionary-style and object-style facts
    
    Args:
        item: Dictionary or object
        key: Attribute/key name
        default: Default value if not found
        
    Returns:
        Value from item or default
        
    Design: Robust accessor that works with any data structure
    """
    if isinstance(item, dict):
        return item.get(key, default)
    else:
        return getattr(item, key, default)


def print_connection_graph(connections: List[Union[Dict, Any]], target: str):
    """
    Print ASCII connection graph.
    
    Args:
        connections: List of connection dicts/objects
        target: Target entity name
        
    Design: Visualization of entity relationships
    """
    print_section_header("🕸️  CONNECTION MAP", "─")
    
    if not connections:
        print("   No connections mapped yet.\n")
        return
    
    # Group by relationship type
    by_type = {}
    for conn in connections:
        rel_type = safe_get(conn, 'relationship_type', 'unknown')
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(conn)
    
    # Display by type
    for rel_type, conns in sorted(by_type.items()):
        print(f"   {rel_type.upper().replace('_', ' ')}")
        print(f"   {'─' * 70}")
        
        for conn in conns[:5]:  # Show max 5 per type
            entity_1 = safe_get(conn, 'entity_1', target)
            entity_2 = safe_get(conn, 'entity_2', 'Unknown')
            strength = safe_get(conn, 'strength', 0.5)
            confidence = safe_get(conn, 'confidence', 0.5)
            
            # Visual strength indicator
            strength_bar = "█" * int(strength * 10)
            confidence_bar = "█" * int(confidence * 10)
            
            print(f"   {entity_1} ←→ {entity_2}")
            print(f"      Strength:   [{strength_bar:<10}] {strength:.2f}")
            print(f"      Confidence: [{confidence_bar:<10}] {confidence:.2f}")
        
        if len(conns) > 5:
            print(f"   ... and {len(conns) - 5} more")
        print()


def print_risk_summary(risks: List[Union[Dict, Any]]):
    """
    Print colorful risk summary.
    
    Args:
        risks: List of risk flag dicts/objects
        
    Design: Clear visualization of risk assessment
    """
    print_section_header("⚠️  RISK ASSESSMENT", "─")
    
    if not risks:
        print("   ✅ No significant risks identified.\n")
        return
    
    # Group by severity
    by_severity = {"critical": [], "high": [], "medium": [], "low": []}
    for risk in risks:
        severity = safe_get(risk, 'severity', 'low')
        by_severity[severity].append(risk)
    
    # Display counts
    print(f"   🔴 Critical: {len(by_severity['critical'])}")
    print(f"   🟠 High:     {len(by_severity['high'])}")
    print(f"   🟡 Medium:   {len(by_severity['medium'])}")
    print(f"   🟢 Low:      {len(by_severity['low'])}\n")
    
    # Show each risk
    for severity in ["critical", "high", "medium", "low"]:
        if by_severity[severity]:
            symbol = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}[severity]
            
            for risk in by_severity[severity][:3]:  # Max 3 per severity
                category = safe_get(risk, 'category', 'unknown').upper()
                desc = safe_get(risk, 'description', 'No description')
                confidence = safe_get(risk, 'confidence', 0.5) * 100
                impact = safe_get(risk, 'impact_score', 5)
                
                print(f"   {symbol} [{category}] {desc}")
                print(f"      Confidence: {confidence:.0f}% | Impact: {impact}/10\n")


def print_fact_categories(facts: List[Union[Dict, Any]]):
    """
    Print fact distribution by category.
    
    Args:
        facts: List of fact dicts/objects
        
    Design: Visual breakdown of research coverage
    """
    print_section_header("📊 FACT DISTRIBUTION", "─")
    
    if not facts:
        print("   No facts discovered yet.\n")
        return
    
    # Count by category
    by_category = {}
    for fact in facts:
        category = safe_get(fact, 'category', 'unknown')
        by_category[category] = by_category.get(category, 0) + 1
    
    # Sort by count
    sorted_cats = sorted(by_category.items(), key=lambda x: x[1], reverse=True)
    
    # Find max count for scaling
    max_count = max(by_category.values()) if by_category else 1
    
    # Display with bars
    for category, count in sorted_cats:
        bar_length = int((count / max_count) * 40)
        bar = "█" * bar_length
        percent = (count / len(facts)) * 100
        
        print(f"   {category:15} {bar:<40} {count:3} ({percent:.1f}%)")
    
    print(f"\n   Total: {len(facts)} facts discovered\n")


def print_top_facts(facts: List[Union[Dict, Any]], limit: int = 10):
    """
    Print top N facts by confidence.
    
    CRITICAL FIX: Properly handles both dict and object-style facts
    
    Args:
        facts: List of fact dicts/objects
        limit: Number of facts to show
        
    Design: Sorted by confidence with clear indicators
    """
    print_section_header("✨ TOP FACTS (Highest Confidence)", "─")
    
    if not facts:
        print("   No facts found.\n")
        print("   ⚠️ Debugging info:")
        print(f"   - Facts list is empty or None")
        print(f"   - This may indicate a research or extraction issue\n")
        return
    
    # Sort by confidence
    sorted_facts = sorted(
        facts,
        key=lambda f: safe_get(f, 'confidence', 0),
        reverse=True
    )
    
    for i, fact in enumerate(sorted_facts[:limit], 1):
        # CRITICAL FIX: Check multiple possible keys for content
        content = (
            safe_get(fact, 'content') or 
            safe_get(fact, 'fact') or 
            safe_get(fact, 'text') or 
            'No content available'
        )
        
        category = safe_get(fact, 'category', 'unknown')
        confidence = safe_get(fact, 'confidence', 0) * 100  # Convert to percentage
        
        # Confidence indicator
        if confidence >= 80:
            indicator = "🟢 HIGH"
        elif confidence >= 60:
            indicator = "🟡 MED"
        else:
            indicator = "🔴 LOW"
        
        print(f"    {i:2}. [{category}] {content}")
        print(f"       {indicator} ({confidence:.1f}%)\n")


# ==============================================================================
# HTML REPORT GENERATION
# ==============================================================================

def generate_html_report(
    result: Dict[str, Any],
    query: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate beautiful, interactive HTML report.
    
    Features:
    - Responsive design
    - Searchable facts
    - Collapsible sections
    - Print-friendly
    - Professional styling
    
    Args:
        result: Research results dictionary
        query: Target entity name
        output_path: Optional output file path
        
    Returns:
        Path to generated HTML file
        
    Design: Self-contained HTML with embedded CSS/JS
    """
    
    if output_path is None:
        # Auto-generate path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
        safe_query = safe_query.replace(' ', '_').strip('_')
        output_path = f"research_report_{safe_query}_{timestamp}.html"
    
    # Extract data
    facts = result.get('facts', [])
    connections = result.get('connections', [])
    risks = result.get('risk_flags', [])
    
    # Calculate score if available
    metadata = result.get('metadata', {})
    score = metadata.get('quality_score', 0)
    grade = metadata.get('quality_grade', 'N/A')
    
    # Generate facts HTML
    facts_html = ""
    for i, fact in enumerate(facts, 1):
        content = (
            safe_get(fact, 'content') or 
            safe_get(fact, 'fact') or 
            safe_get(fact, 'text') or 
            'No content'
        )
        category = safe_get(fact, 'category', 'unknown')
        confidence = safe_get(fact, 'confidence', 0) * 100
        
        conf_class = "fact-high" if confidence >= 80 else "fact-med" if confidence >= 60 else "fact-low"
        
        facts_html += f"""
        <div class="fact {conf_class}">
            <div class="fact-header">
                <strong>[{category.upper()}]</strong>
                <span class="confidence">Confidence: {confidence:.1f}%</span>
            </div>
            <div class="fact-content">{content}</div>
        </div>
        """
    
    # Generate connections HTML
    connections_html = ""
    if connections:
        for i, conn in enumerate(connections, 1):
            entity_1 = safe_get(conn, 'entity_1', query)
            entity_2 = safe_get(conn, 'entity_2', 'Unknown')
            rel_type = safe_get(conn, 'relationship_type', 'unknown')
            strength = safe_get(conn, 'strength', 0.5)
            confidence = safe_get(conn, 'confidence', 0.5)
            
            connections_html += f"""
            <div class="connection">
                <div class="conn-entities">{entity_1} ←→ {entity_2}</div>
                <div class="conn-details">
                    Type: {rel_type} | Strength: {strength:.2f} | Confidence: {confidence:.2f}
                </div>
            </div>
            """
    else:
        connections_html = "<p class='no-data'>No connections mapped</p>"
    
    # Generate risks HTML
    risks_html = ""
    if risks:
        for i, risk in enumerate(risks, 1):
            category = safe_get(risk, 'category', 'unknown').upper()
            severity = safe_get(risk, 'severity', 'low').upper()
            desc = safe_get(risk, 'description', 'No description')
            confidence = safe_get(risk, 'confidence', 0.5) * 100
            
            severity_class = f"risk-{severity.lower()}"
            
            risks_html += f"""
            <div class="risk {severity_class}">
                <div class="risk-header">
                    <strong>[{category}]</strong>
                    <span class="severity">{severity}</span>
                </div>
                <div class="risk-desc">{desc}</div>
                <div class="risk-conf">Confidence: {confidence:.0f}%</div>
            </div>
            """
    else:
        risks_html = "<p class='no-data'>✅ No significant risks identified</p>"
    
    # HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {target}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header h2 {{
            font-size: 1.8em;
            font-weight: 300;
            margin-bottom: 20px;
        }}
        
        .header .meta {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .score-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 25px;
            margin-top: 15px;
            font-size: 1.1em;
        }}
        
        .search-container {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .search-box {{
            width: 100%;
            padding: 12px 20px;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #764ba2;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .section {{
            margin: 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .collapsible {{
            width: 100%;
            padding: 20px;
            background: #f8f9fa;
            border: none;
            text-align: left;
            cursor: pointer;
            font-size: 18px;
            font-weight: 600;
            color: #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }}
        
        .collapsible:hover {{
            background: #e9ecef;
        }}
        
        .collapsible.active {{
            background: #667eea;
            color: white;
        }}
        
        .collapsible::after {{
            content: '▼';
            font-size: 14px;
            transition: transform 0.3s;
        }}
        
        .collapsible.active::after {{
            transform: rotate(180deg);
        }}
        
        .content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: white;
        }}
        
        .content.active {{
            max-height: 5000px;
            padding: 20px;
        }}
        
        .fact {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            transition: transform 0.2s;
        }}
        
        .fact:hover {{
            transform: translateX(5px);
        }}
        
        .fact-high {{ border-left-color: #10b981; }}
        .fact-med {{ border-left-color: #f59e0b; }}
        .fact-low {{ border-left-color: #ef4444; }}
        
        .fact-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .fact-content {{
            color: #333;
            line-height: 1.5;
        }}
        
        .confidence {{
            font-size: 0.85em;
            color: #666;
            font-weight: normal;
        }}
        
        .connection {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}
        
        .conn-entities {{
            font-weight: 600;
            margin-bottom: 5px;
            color: #333;
        }}
        
        .conn-details {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .risk {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid;
            border-radius: 4px;
        }}
        
        .risk-critical {{ border-left-color: #dc2626; }}
        .risk-high {{ border-left-color: #ea580c; }}
        .risk-medium {{ border-left-color: #f59e0b; }}
        .risk-low {{ border-left-color: #10b981; }}
        
        .risk-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }}
        
        .severity {{
            background: rgba(0,0,0,0.1);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
        
        .risk-desc {{
            margin: 10px 0;
            color: #333;
        }}
        
        .risk-conf {{
            font-size: 0.85em;
            color: #666;
        }}
        
        .no-data {{
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
            .search-container {{ display: none; }}
            .collapsible {{ background: white; border-bottom: 2px solid #333; }}
            .content {{ max-height: none !important; padding: 20px; }}
        }}
        
        @media (max-width: 768px) {{
            .header {{ padding: 20px; }}
            .header h1 {{ font-size: 1.8em; }}
            .header h2 {{ font-size: 1.3em; }}
            .container {{ border-radius: 0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Deep Research Report</h1>
            <h2>{target}</h2>
            <div class="meta">
                <div>Generated: {date}</div>
                <div class="score-badge">Quality Score: {score}/100 - Grade {grade}</div>
            </div>
        </div>
        
        <div class="search-container">
            <input type="text" class="search-box" id="searchInput" 
                   placeholder="🔍 Search facts, connections, and risks..." 
                   onkeyup="searchContent()">
        </div>
        
        <div class="section">
            <button class="collapsible" onclick="toggleSection(this)">
                📋 All Facts ({fact_count})
            </button>
            <div class="content">
                {facts_html}
            </div>
        </div>
        
        <div class="section">
            <button class="collapsible" onclick="toggleSection(this)">
                🕸️ Connections ({connection_count})
            </button>
            <div class="content">
                {connections_html}
            </div>
        </div>
        
        <div class="section">
            <button class="collapsible" onclick="toggleSection(this)">
                ⚠️ Risk Flags ({risk_count})
            </button>
            <div class="content">
                {risks_html}
            </div>
        </div>
    </div>
    
    <script>
        // Toggle section visibility
        function toggleSection(button) {{
            button.classList.toggle('active');
            const content = button.nextElementSibling;
            content.classList.toggle('active');
        }}
        
        // Search functionality
        function searchContent() {{
            const input = document.getElementById('searchInput').value.toLowerCase();
            const allItems = document.querySelectorAll('.fact, .connection, .risk');
            
            allItems.forEach(item => {{
                const text = item.textContent.toLowerCase();
                item.style.display = text.includes(input) ? 'block' : 'none';
            }});
        }}
        
        // Auto-expand first section on load
        window.addEventListener('load', () => {{
            const firstButton = document.querySelector('.collapsible');
            if (firstButton) {{
                toggleSection(firstButton);
            }}
        }});
    </script>
</body>
</html>"""
    
    # Fill template
    html = html_template.format(
        target=query,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        score=score,
        grade=grade,
        fact_count=len(facts),
        facts_html=facts_html,
        connection_count=len(connections),
        connections_html=connections_html,
        risk_count=len(risks),
        risks_html=risks_html
    )
    
    # Save file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    # Get absolute path
    abs_path = os.path.abspath(output_path)
    
    logger.info(f"HTML report generated: {abs_path}")
    
    return abs_path


# ==============================================================================
# MAIN DEMO FUNCTION
# ==============================================================================

async def run_demo(
    query: str,
    max_iterations: int = 3,
    show_full: bool = False,
    save_results: bool = False,
    generate_html: bool = False
):
    """
    Run research demo with beautiful output.
    
    FAANG-Quality Features:
    - Comprehensive error handling
    - Beautiful ASCII output
    - Optional HTML report generation
    - JSON export capability
    - Production-ready logging
    
    Args:
        query: Target to research
        max_iterations: Number of search iterations
        show_full: Show all facts/connections or just summary
        save_results: Save results to JSON file
        generate_html: Generate interactive HTML report
        
    Returns:
        Research results dictionary
        
    Design: User-friendly interface with multiple output formats
    """
    
    print(BANNER)
    
    # Display settings
    print("🎯 RESEARCH TARGET")
    print(f"   Subject:     {query}")
    print(f"   Iterations:  {max_iterations}")
    print(f"   Full Output: {'Yes' if show_full else 'No (use --full flag)'}")
    print(f"   Save Results: {'Yes' if save_results else 'No (use --save flag)'}")
    print(f"   HTML Report: {'Yes' if generate_html else 'No (use --html flag)'}\n")
    
    # Initialize
    print("🔧 Initializing AI models...")
    
    try:
        orchestrator = ResearchOrchestrator(
            max_iterations=max_iterations,
            enable_checkpoints=False
        )
        print("✅ Ready!\n")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
        return None
    
    # Start research
    print_section_header("🔍 STARTING RESEARCH", "═")
    print(f"   Target: {query}")
    print(f"   This will take approximately {max_iterations * 20}-{max_iterations * 30} seconds...\n")
    
    start_time = datetime.now()
    
    try:
        # Run research
        result = await orchestrator.research(
            target_name=query,
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Extract results
        facts = result.get('facts', [])
        risks = result.get('risk_flags', [])
        connections = result.get('connections', [])
        coverage = result.get('coverage', {})
        
        # ==================================================================
        # DISPLAY RESULTS
        # ==================================================================
        
        print_section_header("✅ RESEARCH COMPLETE", "═")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Facts:    {len(facts)}")
        print(f"   Risks:    {len(risks)}")
        print(f"   Connections: {len(connections)}\n")
        
        # Fact distribution
        print_fact_categories(facts)
        
        # Risk assessment
        print_risk_summary(risks)
        
        # Connection map
        print_connection_graph(connections, query)
        
        # Top facts
        print_top_facts(facts, limit=len(facts) if show_full else 10)
        
        # ==================================================================
        # FULL OUTPUT (if requested)
        # ==================================================================
        
        if show_full and facts:
            print_section_header("📋 ALL FACTS (DETAILED)", "═")
            
            for i, fact in enumerate(facts, 1):
                content = (
                    safe_get(fact, 'content') or 
                    safe_get(fact, 'fact') or 
                    safe_get(fact, 'text') or 
                    'No content'
                )
                category = safe_get(fact, 'category', 'unknown')
                confidence = safe_get(fact, 'confidence', 0)
                verified = safe_get(fact, 'verified', False)
                
                status = "✓ VERIFIED" if verified else "○ UNVERIFIED"
                
                print(f"\n   {i:3}. [{category.upper()}] {content}")
                print(f"        Confidence: {confidence:.1%} | {status}")
            
            print()
        
        # ==================================================================
        # SAVE RESULTS (if requested)
        # ==================================================================
        
        if save_results:
            output_dir = project_root / "research_results"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
            safe_query = safe_query.replace(' ', '_').strip('_')
            filename = f"{safe_query}_{timestamp}.json"
            filepath = output_dir / filename
            
            # Convert facts to dicts for JSON serialization
            facts_dicts = []
            for fact in facts:
                if isinstance(fact, dict):
                    facts_dicts.append(fact)
                else:
                    facts_dicts.append({
                        'content': safe_get(fact, 'content', ''),
                        'category': safe_get(fact, 'category', ''),
                        'confidence': safe_get(fact, 'confidence', 0),
                        'verified': safe_get(fact, 'verified', False),
                        'source_urls': safe_get(fact, 'source_urls', []),
                    })
            
            output = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'summary': {
                    'facts_count': len(facts),
                    'risk_flags_count': len(risks),
                    'connections_count': len(connections),
                    'iterations': max_iterations,
                },
                'facts': facts_dicts,
                'risk_flags': risks,
                'connections': connections,
                'coverage': coverage,
            }
            
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            
            print_section_header("💾 RESULTS SAVED", "─")
            print(f"   File: {filepath}")
            print(f"   Size: {filepath.stat().st_size / 1024:.1f} KB\n")
        
        # ==================================================================
        # GENERATE HTML REPORT (if requested)
        # ==================================================================
        
        if generate_html:
            try:
                html_path = generate_html_report(result, query)
                
                print_section_header("🌐 HTML REPORT GENERATED", "─")
                print(f"   File: {html_path}")
                print(f"   Opening in browser...\n")
                
                # Open in browser
                webbrowser.open(f'file://{html_path}')
                
            except Exception as e:
                print(f"   ❌ Failed to generate HTML report: {e}")
                logger.error(f"HTML generation failed: {e}", exc_info=True)
        
        # ==================================================================
        # INTERACTIVE OPTIONS
        # ==================================================================
        
        print_section_header("💡 WHAT'S NEXT?", "─")
        
        if not show_full:
            print("   1. Re-run with --full flag to see ALL facts and connections")
        if not save_results:
            print("   2. Re-run with --save flag to save results to JSON")
        if not generate_html:
            print("   3. Re-run with --html flag to generate interactive HTML report")
        print(f"   4. Re-run with --iterations 5 for deeper research")
        print(f"   5. Try another person: python scripts/demo.py \"Elon Musk\"")
        print()
        
        print_section_header("✅ DEMO COMPLETE", "═")
        
        return result
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    """
    CLI entry point with comprehensive argument parsing.
    
    Design: User-friendly interface with validation
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Deep Research AI Agent - Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo.py "Tim Cook"                           # Quick summary
  python scripts/demo.py "Tim Cook" --full                    # Show everything
  python scripts/demo.py "Tim Cook" --iterations 5            # More thorough
  python scripts/demo.py "Tim Cook" --save                   # Save JSON
  python scripts/demo.py "Tim Cook" --html                   # Generate HTML
  python scripts/demo.py "Tim Cook" --full --save --html     # Complete output

FAANG-Quality Features:
  • Beautiful ASCII visualizations
  • Interactive HTML reports
  • Robust error handling
  • Multiple output formats
  • Production-ready code
        """
    )
    
    parser.add_argument(
        "query",
        help="Person or entity to research"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of search iterations (default: 3)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show ALL facts and connections (not just summary)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate interactive HTML report and open in browser"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Deep Research AI Agent Demo v3.0"
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.query or not args.query.strip():
        parser.error("Query cannot be empty")
    
    if args.iterations < 1 or args.iterations > 20:
        parser.error("Iterations must be between 1 and 20")
    
    # Run demo
    result = asyncio.run(run_demo(
        query=args.query.strip(),
        max_iterations=args.iterations,
        show_full=args.full,
        save_results=args.save,
        generate_html=args.html
    ))
    
    sys.exit(0 if result else 1)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)