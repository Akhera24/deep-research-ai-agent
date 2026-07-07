#!/usr/bin/env python3
"""
Research Script - Full Research with Comprehensive Output

Usage examples:
    python scripts/research.py "Tim Cook"
    python scripts/research.py "Satya Nadella" --save
    python scripts/research.py "Jensen Huang" --iterations 15 --save --html
    python scripts/research.py "Elon Musk" -i 20 -s --output results/elon.json

Features:
- Full research with all facts
- Detailed breakdown by category
- Risk flags and connections
- Source validation
- Optional save to JSON
- HTML report generation
- Comprehensive error handling
"""

import sys
import json
import time
import asyncio
import os
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Setup Python path BEFORE any imports
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

print(f"📁 Project root: {project_root}")
print("🐍 Python path configured\n")

# Import after path setup
try:
    from src.core.workflow import ResearchOrchestrator
    from config.logging_config import get_logger
    from config.settings import validate_settings
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root or scripts directory")
    print("Required: src/, config/, scripts/ directories")
    sys.exit(1)

logger = get_logger(__name__)


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_header(text: str, char: str = "=") -> str:
    """Format a header with decorative lines (80 chars)"""
    line = char * 80
    return f"\n{line}\n{text}\n{line}\n"


def format_section(text: str) -> str:
    """Format a section header (80 chars)"""
    return f"\n{text}\n{'-' * 80}"

# ============================================================================
# QUALITY SCORE + HTML REPORT (moved to src/reporting/html_report.py —
# shared with the job API; escaping applied on render)
# ============================================================================

from src.reporting.html_report import calculate_quality_score, generate_html_report


# ============================================================================
# MAIN RESEARCH FUNCTION (ASYNC)
# ============================================================================

async def run_research_async(
    query: str,
    iterations: int = 10,
    save: bool = False,
    generate_html: bool = False,
    output_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Run comprehensive research on a person (async implementation)."""
    
    print(format_header("🔍 DEEP RESEARCH AI AGENT - FULL RESEARCH"))
    
    print(f"Query:          {query}")
    print(f"Max Iterations: {iterations}")
    print(f"Save Results:   {save}")
    print(f"HTML Report:    {generate_html}")
    
    # Validate settings
    print(format_section("🔧 INITIALIZATION"))
    try:
        validate_settings()
        print("✅ Settings validated")
    except Exception as e:
        print(f"❌ Settings validation failed: {e}")
        return None
    
    # Initialize orchestrator
    try:
        orchestrator = ResearchOrchestrator(
            max_iterations=iterations,
            enable_checkpoints=False
        )
        print("✅ Orchestrator initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None
    
    # Run research
    print(format_section(f"🔍 RESEARCHING: {query}"))
    print("Running autonomous research...\n")
    
    start_time = time.time()
    
    try:
        result = await orchestrator.research(
            target_name=query,
            context=None
        )
        
        duration = time.time() - start_time
        
        # Display results
        display_results(result, query, duration)
        
        # Save JSON
        if save:
            save_path = output_path or generate_output_path(query)
            save_results(query, result, duration, save_path)
        
        # Generate HTML
        if generate_html:
            try:
                html_path = generate_html_report(result, query, duration)
                print(f"\n🌐 HTML Report Generated: {html_path}")
                print("   Opening in browser...")
                webbrowser.open(f'file://{html_path}')
            except Exception as e:
                print(f"   ❌ HTML generation failed: {e}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        return None


# ============================================================================
# RESULT DISPLAY 
# ============================================================================

def display_results(result: Dict[str, Any], query: str, duration: float):
    """Display comprehensive research results."""
    
    print(format_header("📊 COMPREHENSIVE RESEARCH RESULTS"))
    
    # Extract data
    facts = result.get('facts', [])
    risk_flags = result.get('risk_flags', [])
    connections = result.get('connections', [])
    metadata = result.get('metadata', {})
    
    # Metadata
    iterations_used = metadata.get('iterations', 0)
    queries_executed = metadata.get('queries_executed', 0)
    coverage = metadata.get('coverage', {})  # ✅ Get from metadata
    total_cost = metadata.get('total_cost', 0.0)
    
    # Summary
    print(format_section("📈 SUMMARY STATISTICS"))
    print(f"  Subject:           {query}")
    print(f"  Facts Found:       {len(facts):,}")
    print(f"  Risk Flags:        {len(risk_flags)}")
    print(f"  Connections:       {len(connections)}")
    print(f"  Searches:          {queries_executed}")
    print(f"  Iterations:        {iterations_used}")
    print(f"  Duration:          {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Cost:              ${total_cost:.2f}")
    
    # Coverage
    if coverage:
        print(format_section("📊 RESEARCH COVERAGE"))
        for category, score in sorted(coverage.items(), key=lambda x: x[1], reverse=True):
            if category != 'average':
                percentage = score * 100
                bar_length = int(percentage / 2)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"  {category:.<20} [{bar}] {percentage:>5.1f}%")
        
        if 'average' in coverage:
            print(f"\n  {'Overall Coverage':.<20} {coverage['average'] * 100:.1f}%")
    
    # Facts by category
    if facts:
        categories = {}
        for fact in facts:
            cat = fact.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(format_section("📂 FACTS BY CATEGORY"))
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(facts)) * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {category:.<20} {count:>3} [{bar}] {percentage:>5.1f}%")
    
    # Risk flags
    if risk_flags:
        print(format_section("⚠️  RISK FLAGS IDENTIFIED"))
        for i, flag in enumerate(risk_flags, 1):
            print(f"\n  {i}. [{flag.get('category', 'unknown').upper()}] {flag.get('description', '')}")
            print(f"     Severity: {flag.get('severity', 'unknown').upper()} | Confidence: {flag.get('confidence', 0):.1%}")
    
    # Connections
    if connections:
        print(format_section("🕸️  CONNECTIONS & RELATIONSHIPS"))
        for i, conn in enumerate(connections[:5], 1):
            print(f"\n  {i}. {conn.get('entity_1', query)} ↔ {conn.get('entity_2', 'Unknown')}")
            print(f"     Type: {conn.get('relationship_type', 'unknown')} | Strength: {conn.get('strength', 0):.2f}")
    
    # Final summary
    print(format_header("✅ RESEARCH COMPLETE", "="))
    
    # Pass coverage from metadata, not from result root
    if facts:
        score_result = calculate_quality_score(
            facts=facts,
            risk_flags=risk_flags,
            connections=connections,
            coverage=coverage  # Use coverage from metadata
        )
        
        quality_score = score_result['score']
        grade = f"{score_result['grade']} ({score_result['quality']})"
        indicator = score_result['indicator']
        
        print(f"\n{indicator} Research Quality Score: {quality_score:.1f}/100 - Grade {grade}")
        
        if score_result['strengths']:
            print("\n💪 Strengths:")
            for strength in score_result['strengths']:
                print(f"  {strength}")
        
        if score_result['recommendations']:
            print("\n💡 Recommendations for Improvement:")
            for rec in score_result['recommendations']:
                print(f"  {rec}")
        
        print("\n\U0001f4ca Score Breakdown:")
        print(f"  Fact Quality:       {score_result['breakdown']['fact_quality']}/35")
        print(f"  Coverage:           {score_result['breakdown']['coverage']}/25")
        print(f"  Risk Assessment:    {score_result['breakdown']['risk_assessment']}/20")
        print(f"  Connection Mapping: {score_result['breakdown']['connection_mapping']}/20")


# ============================================================================
# FILE PERSISTENCE
# ============================================================================

def generate_output_path(query: str, output_dir: str = "research_results") -> Path:
    """Generate output file path with timestamp."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
    safe_query = safe_query.replace(' ', '_').strip('_')
    filename = f"{safe_query}_{timestamp}.json"
    
    return output_path / filename


def save_results(query: str, result: Dict[str, Any], duration: float, filepath: Path) -> bool:
    """Save research results to JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "metadata": {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "version": "3.0",
                "agent": "Deep Research AI Agent"
            },
            "summary": {
                "facts_count": len(result.get('facts', [])),
                "risk_flags_count": len(result.get('risk_flags', [])),
                "connections_count": len(result.get('connections', [])),
                "total_cost": result.get('metadata', {}).get('total_cost', 0),
                "iterations": result.get('metadata', {}).get('iterations', 0)
            },
            "results": result
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Results saved successfully!")
        print(f"   File: {filepath}")
        print(f"   Size: {filepath.stat().st_size / 1024:.1f} KB")
        print(f"   ✅ File validated")
        
        return True
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")
        return False


# ============================================================================
# SYNCHRONOUS WRAPPER
# ============================================================================

def run_research(
    query: str,
    iterations: int = 10,
    save: bool = False,
    generate_html: bool = False,
    output_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Run research (synchronous wrapper)."""
    try:
        result = asyncio.run(run_research_async(
            query=query,
            iterations=iterations,
            save=save,
            generate_html=generate_html,
            output_path=output_path
        ))
        return result
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return None


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point with comprehensive argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Deep Research AI Agent - Full Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/research.py "Tim Cook"
  python scripts/research.py "Satya Nadella" --save
  python scripts/research.py "Jensen Huang" --iterations 15 --save --html
  python scripts/research.py "Elon Musk" -i 20 -s --html

Features:
  • Comprehensive fact extraction
  • Risk pattern recognition
  • Connection mapping
  • Source validation
  • JSON export
  • HTML report generation
        """
    )
    
    parser.add_argument('query', help='Person to research')
    parser.add_argument('-i', '--iterations', type=int, default=10, help='Max iterations (default: 10)')
    parser.add_argument('-s', '--save', action='store_true', help='Save results to JSON')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--output', type=str, help='Specific output file path')
    parser.add_argument('--version', action='version', version='Deep Research AI Agent v3.0')
    
    args = parser.parse_args()
    
    # Validate
    if not args.query or not args.query.strip():
        parser.error("Query cannot be empty")
    if args.iterations < 1 or args.iterations > 50:
        parser.error("Iterations must be between 1 and 50")
    
    # Run research
    result = run_research(
        query=args.query.strip(),
        iterations=args.iterations,
        save=args.save,
        generate_html=args.html,
        output_path=Path(args.output) if args.output else None
    )
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)